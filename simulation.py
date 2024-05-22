import numpy as np
from tqdm import tqdm
from numba import cuda

from analysis import SimulationResult
from model import Model
from sources import Source
from filtering import lp_filter, lp_filter2, lp_butter_filter

from cuda_kernels_2D import (
    propagation_kernel_2D,
    calc_dphi_dpsi_kernel_2D,
    update_phi_psi_kernel_2D,
    exchange_p_kernel_2D,
)


class FDTD_sim:
    def __init__(
        self, model: Model, subsample_space: int = 1, subsample_time: int = 1
    ) -> None:
        """
        Initializes a Finite-Difference Time-Domain (FDTD) simulation for wave propagation.

        Parameters:
        - model (Model): An instance of the Model class representing the simulation parameters.

        Attributes:
        - model (Model): An instance of the Model class representing the simulation parameters.
        - sources (list): A list to store sources added to the simulation.
        - time_order (int): The order of the temporal scheme used in the simulation.

        """
        self.model = model
        self.sources = []
        self.time_order = 6
        self.subsample_space = subsample_space
        self.subsample_time = subsample_time
        self.init_p = None
        self.it0 = 0
        if subsample_time > subsample_space:
            print(
                "Warning: subsample_time is larger than subsample_space. Domain extension may not be possible."
            )

    def add_source(self, source: Source):
        """
        Adds a source to the simulation.

        Parameters:
        - source: An object representing the source to be added.

        """
        if source.max_freq > 2 / self.model.dt:
            print(
                f"WARNING! Model dt: ({self.model.dt}) too large for source frequency ({source.max_freq})!"
            )
        self.sources.append(source)

    def set_initial_pressure(self, p: np.ndarray):
        # if p.shape[1] != self.model.nr or p.shape[2] != self.model.nz:
        #     raise ValueError("Initial pressure field must be of size (nt_init, nr, nz)")
        self.init_p = p
        self.it0 = self.init_p.shape[0]

    def propagate_cuda(self, type: str = "linear") -> np.ndarray:
        """
        Propagates the wave field over multiple time steps using the specified propagation type.
        This is run on the GPU using CUDA.

        Parameters:
        - p (numpy.ndarray): Initial pressure field.
        - type (str): Type of propagation ("linear" or "nonlinear").

        Returns:
        - numpy.ndarray: Saved pressure fields over time.

        """
        if type not in ["linear", "nonlinear"]:
            raise ValueError(f'Invalid type: {type} not in ("linear", "nonlinear")')

        # Unpack model attributes
        nt = self.model.nt
        nr = self.model.nr
        nz = self.model.nz
        dr = self.model.dr
        dz = self.model.dz
        dt = self.model.dt
        c0 = self.model.c0

        pml_width = self.model.pml_width
        domain_slice_r = slice(2, nr - pml_width, self.subsample_space)
        domain_slice_z = slice(pml_width, nz - pml_width, self.subsample_space)
        # domain_slice_r = slice(0, None, self.subsample_space)
        # domain_slice_z = slice(0, None, self.subsample_space)
        if pml_width == 0:
            domain_slice_r = slice(2, -2, self.subsample_space)
            domain_slice_z = slice(2, -2, self.subsample_space)

        # Initialize arrays
        p_saved = np.zeros(
            (
                nt // self.subsample_time + 1,
                np.ceil(self.model.nrd / self.subsample_space).astype(np.int32),
                np.ceil(self.model.nzd / self.subsample_space).astype(np.int32),
            ),
            dtype=np.float32,
        )
        p = np.zeros((5, nr, nz))

        if self.init_p is not None:
            p_saved[
                : self.init_p.shape[0] // self.subsample_time,
                : np.ceil(self.init_p.shape[1] / self.subsample_space),
                : np.ceil(self.init_p.shape[2] / self.subsample_space),
            ] = self.init_p[
                :: self.subsample_time, :: self.subsample_space, :: self.subsample_space
            ]
            p[
                1:,
                2 : self.init_p.shape[1] + 2,
                pml_width : self.init_p.shape[2] + pml_width,
            ] = self.init_p[-4:][::-1]

        # [phir, phiz, psir, psiz, dphir, dphiz, dpsir, dpsiz]
        phipsi = np.zeros((8, nr, nz))

        # Copy to device
        p_d = cuda.to_device(p)
        phipsi_d = cuda.to_device(phipsi)
        sigma_d = cuda.to_device(self.model.sigma)

        # Setup cuda parameters
        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(nr / threadsperblock[0]))
        blockspergrid_z = int(np.ceil(nz / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_z)

        # Pack variables
        model_data = (
            nr,
            nz,
            dr,
            dz,
            dt,
            c0,
            self.model.beta,
            self.model.rho0,
            self.model.diffusivity,
        )

        # Initialize source
        source = self.sources[0]
        source_response_d = cuda.to_device(source.response)
        source_profile_d = cuda.to_device(source.profile)
        source_phase_profile_d = cuda.to_device(source.phase_profile)
        source_data = (
            source_response_d,
            source_profile_d,
            source_phase_profile_d,
            source.ir,
            source.iz,
            source.iwidth,
        )

        # Initialize stream
        stream = cuda.stream()

        p_current = np.zeros((nr, nz))

        for it in tqdm(range(self.it0, nt)):
            if it % self.subsample_time == 0:
                # Copy previous pressure field to host asynchonously
                p_d[1].copy_to_host(p_current, stream=stream)

            propagation_kernel_2D[blockspergrid, threadsperblock](
                p_d,
                model_data,
                source_data,
                phipsi_d,
                sigma_d,
                it,
                True if type == "nonlinear" else False,
            )

            calc_dphi_dpsi_kernel_2D[blockspergrid, threadsperblock](
                p_d, model_data, phipsi_d, sigma_d
            )

            update_phi_psi_kernel_2D[blockspergrid, threadsperblock](
                model_data, phipsi_d
            )

            cuda.synchronize()

            if it % self.subsample_time == 0:
                # Save iteration only inside domain, one timestep behind due to asynchonous copy
                p_saved[it // self.subsample_time - 1] = p_current[
                    domain_slice_r, domain_slice_z
                ]

            # Exchange timesteps
            exchange_p_kernel_2D[blockspergrid, threadsperblock](model_data, p_d)

        # Copy the last timestep to host
        p_d[1].copy_to_host(p_current)
        p_saved[-1] = p_current[domain_slice_r, domain_slice_z]

        return p_saved

    def run_simulation_cuda(
        self,
        type: str = "linear",
    ) -> SimulationResult:

        if type == "linear":
            p_saved = self.propagate_cuda(type="linear")
        elif type == "nonlinear":
            p_saved = self.propagate_cuda(type="nonlinear")
        else:
            raise ValueError(f"Invalid type: {type} not in ('linear', 'nonlinear')")

        result = SimulationResult(
            p_saved,
            self.model,
            self.sources,
            subsample_space=self.subsample_space,
            subsample_time=self.subsample_time,
        )
        return result


def extend_domain(
    result_init: SimulationResult,
    extend_from: int,
    depth_multiplier: float,
    width_multiplier: float,
    cutoff: float = -1,
    additional_subsampling: int = 1,
):
    """
    Given a simulation_result, extend the domain by subsampling, filtering and propagating linearly.
    Make sure that the initial wave propagated the full domain.

    Args:
        simulation_result (SimulationResult): SimulationResult object.
        extend_from (int): Time index from which to extend propagation.
        depth_multiplier (float): Factor by which to extend depth.
        width_multiplier (float): Factor by which to extend width.
        cutoff (float, optional): Cutoff frequency for the filter. Defaults to -1.

    Returns:
        SimulationResult: Final simulation result.
    """
    if cutoff == -1:
        cutoff = 2 * abs(
            result_init.sources[0].frequency2 - result_init.sources[0].frequency1
        )

    # # Filter and subsample time
    # p_filtered_subsampled = lp_butter_filter(
    #     result_init.p, result_init.model.dt, cutoff=cutoff, order=2
    # )[
    #     :: result_init.subsample_space
    # ]  # TODO implement better filter

    # Filter and subsample time
    p_filtered = lp_filter(
        result_init.p, result_init.model.dt * result_init.subsample_time, cutoff=cutoff
    )
    # p_filtered = result_init.p

    # Create new model
    pml_width = int(2 * result_init.model.pml_width / result_init.subsample_space)

    model_extended = Model(
        nr=p_filtered.shape[1] * width_multiplier + pml_width + 2,
        nz=p_filtered.shape[2] * depth_multiplier + 2 * pml_width,
        nt=p_filtered.shape[0] * depth_multiplier,
        dr=result_init.model.dr * result_init.subsample_space,
        dz=result_init.model.dz * result_init.subsample_space,
        dt=result_init.model.dt * result_init.subsample_time,
        c0=result_init.model.c0,
        beta=result_init.model.beta,
        rho0=result_init.model.rho0,
        diffusivity=result_init.model.diffusivity,
    )

    # Create new PML
    pml_mask = np.zeros((model_extended.nr, model_extended.nz))
    pml_mask[model_extended.nr - pml_width :, :] = 1
    pml_mask[:, :pml_width] = 1
    pml_mask[:, model_extended.nz - pml_width :] = 1
    model_extended.set_pml_mask(pml_mask, sigma_scaling=0, pml_width=pml_width)
    model_extended.sigma *= 10 / pml_width / 4
    model_extended.plot(axis_label="space")

    # Propagate linearly
    new_sim = FDTD_sim(
        model_extended,
        subsample_space=additional_subsampling,
        subsample_time=1,
    )
    new_sim.set_initial_pressure(p_filtered[:extend_from])

    # Add dummy source
    # new_source = Source(0, pml_width, 10, model_extended.nt, 1, 1, 1, 1, 1)
    new_source = result_init.sources[0]
    new_source.scaling_factor *= result_init.subsample_time
    new_source.response *= 0
    new_sim.add_source(new_source)

    result = new_sim.run_simulation_cuda(type="linear")

    return result
