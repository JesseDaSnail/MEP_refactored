import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


class Model:
    def __init__(
        self,
        nr: int,
        nz: int,
        nt: int,
        dr: float,
        dz: float,
        dt: float,
        c0: float,
        beta: float,
        rho0: float,
        diffusivity: float,
    ) -> None:
        """
        Initializes a wave propagation model with the specified parameters.

        Parameters:
        - nr (int): Number of grid points along the r-axis.
        - nz (int): Number of grid points along the z-axis.
        - nt (int): Number of time steps.
        - dr (float): Grid spacing in the r-axis.
        - dz (float): Grid spacing in the z-axis.
        - dt (float): Time step size.
        - c0 (float): Wave velocity in the medium.
        - beta (float): Coefficient of nonlinearity.
        - rho0 (float): Density of the medium.
        - diffusivity (float): Diffusivity parameter for attenuation.
        """

        self.nr = nr
        self.nz = nz
        self.nt = nt
        self.dr = dr
        self.dz = dz
        self.dt = dt
        self.c0 = c0
        self.beta = beta
        self.rho0 = rho0
        self.diffusivity = diffusivity
        self.pml_width = 0
        self.nrd = nr
        self.nzd = nz

        # Test courant number
        courant = 3 * c0 / dr * dt
        if courant >= 1:
            print(
                f"WARNING! Courant number should be smaller than 1. Currently: {courant}. Simulation may be unstable."
            )

        # Test whether the wave propagates far enough
        if c0 * nt * dt < nz * dz:
            print(
                f"WARNING! The wave may not propagate far enough. Currently: {c0 * nt * dt}."
            )

    def set_pml_mask(
        self, pml_mask: np.ndarray, sigma_scaling: float = 0, pml_width: int = 0
    ):
        """
        Sets the PML mask and calculates corresponding damping coefficients.

        Parameters:
        - pml_mask (numpy.ndarray): Initial PML mask.
        - sigma_scaling (float): Scaling factor for the damping coefficients.
        - pml_width (int): Width of the PML region.

        Returns:
        - numpy.ndarray: Updated PML mask.

        """
        if pml_mask.shape[0] != self.nr or pml_mask.shape[1] != self.nz:
            raise ValueError("pml_mask must be the same shape as the model.")

        self.pml_mask = pml_mask
        self.pml_width = pml_width
        self.nrd = self.nr - 1 * pml_width - 2 - 2 * (pml_width == 0)
        self.nzd = self.nz - 2 * pml_width - 4 * (pml_width == 0)

        distances = distance_transform_edt(pml_mask, return_distances=True)
        sigma_mag = self.c0 / self.dr / 10
        self.sigma = (
            sigma_mag
            * self.pml_mask
            * (np.sqrt(2) * distances / max(1, distances.max())) ** sigma_scaling
        )
        self.sigma = np.clip(self.sigma, 0, sigma_mag)

        # Remove 2 layers at the boundaries to prevent instability
        layers_to_remove = np.arange(-2, 2)
        self.pml_mask[layers_to_remove, :] = 0
        self.pml_mask[:, layers_to_remove] = 0

        return pml_mask

    def plot(self, axis_label: str = "space"):
        """
        Plots the damping coefficients (sigma) associated with the PML.

        Parameters:
        - axis_label (str): Label for the axis of the plot ("space" or "index").

        """
        if axis_label not in ["space", "index"]:
            raise ValueError('axis_label must be either "space" or "index".')
        # plt.figure(figsize=(10, 10))
        # plt.rcParams.update({"font.size": 22})
        if axis_label == "space":
            plt.imshow(
                self.sigma[:, :].T,
                extent=[
                    0,
                    (self.nrd + self.pml_width) * self.dr,
                    (self.nzd + self.pml_width) * self.dz,
                    -self.pml_width * self.dz,
                ],
                origin="upper",
            )
            plt.xlabel("r (m)")
            plt.ylabel("z (m)")
        elif axis_label == "index":
            plt.imshow(
                self.sigma[:, :].T,
                extent=[0, self.nr, self.nz, 0],
                origin="upper",
            )
            plt.xlabel("r")
            plt.ylabel("z")
        plt.colorbar()
        plt.title("Sigma")

    def __str__(self) -> str:
        return (
            f"Model:\n"
            f"  nr={self.nr}\n"
            f"  nz={self.nz}\n"
            f"  dr={self.dr} m\n"
            f"  dz={self.dz} m\n"
            f"  nt={self.nt}\n"
            f"  dt={self.dt} s\n"
            f"  domain_r={self.nrd*self.dr} m\n"
            f"  domain_z={self.nzd*self.dz} m\n"
            f"  pml_width={self.pml_width} / {self.pml_width*self.dz} m"
        )
