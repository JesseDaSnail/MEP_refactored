from numba import cuda
import numpy as np


@cuda.jit
def propagation_kernel_2D(
    p: np.ndarray,
    model_data: tuple[int, int, float, float, float, float, float, float, float],
    source_data: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    phipsi: np.ndarray,
    sigma: np.ndarray,
    it: int,
    is_nonlinear: bool,
):
    # Unpack data
    nr, nz, dr, dz, dt, c0, beta, rho0, diffusivity = model_data
    (
        source_response,
        source_profile,
        source_phase_profile,
        source_ir,
        source_iz,
        source_iwidth,
    ) = source_data

    ir, iz = cuda.grid(2)

    if ir > 1 and ir < nr - 2 and iz > 1 and iz < nz - 2:
        # Linear wave equation
        p[0, ir, iz] = (
            laplace_2D(p[1], ir, iz, dr, dz) * c0**2 * dt**2
            + 2 * p[1, ir, iz]
            - p[2, ir, iz]
        )

        if is_nonlinear:
            # Add nonlinearity
            dpdt2 = p[1, ir, iz] ** 2 - 2 * p[2, ir, iz] ** 2 + p[3, ir, iz] ** 2
            nonlinear_term = beta / (rho0 * c0**2) * dpdt2
            p[0, ir, iz] += nonlinear_term

        attenuation = (
            diffusivity
            / c0**2
            * (p[1, ir, iz] - 3 * p[2, ir, iz] + 3 * p[3, ir, iz] - p[4, ir, iz])
            / dt
        )
        p[0, ir, iz] += attenuation

        # Calculate PML term
        sum_term_r = (
            sigma[ir, iz] * phipsi[2, ir + 1, iz]
            - sigma[ir - 1, iz] * phipsi[0, ir - 1, iz]
        ) / dr

        sum_term_z = (
            sigma[ir, iz] * phipsi[3, ir, iz + 1]
            - sigma[ir, iz - 1] * phipsi[1, ir, iz - 1]
        ) / dz

        pml_term = sum_term_r + sum_term_z

        # Add PML term
        p[0, ir, iz] += pml_term * c0**2 * dt**2

    # Inject source
    if ir >= source_ir and ir < source_ir + source_iwidth and iz == source_iz:
        time_index = it + source_phase_profile[ir - source_ir]
        time_index = min(max(0, time_index), len(source_response) - 1)
        p[0, ir, iz] += (
            source_profile[ir - source_ir] * source_response[time_index] * c0**2 * dt**2
        )

    # if iz == 0:
    #     p[0, ir, 0] = p[0, ir, 3]
    # if iz == 1:
    #     p[0, ir, 1] = p[0, ir, 2]
    if ir == 0 and iz > 1 and iz < nz - 2:
        p[0, 0, iz] = p[0, 3, iz]
    if ir == 1 and iz > 1 and iz < nz - 2:
        p[0, 1, iz] = p[0, 2, iz]


@cuda.jit(device=True)
def laplace_2D(p_current: np.ndarray, ir: int, iz: int, dr: float, dz: float):
    ddr2 = (
        (
            -p_current[ir + 2, iz]
            + 16 * p_current[ir + 1, iz]
            - 30 * p_current[ir, iz]
            + 16 * p_current[ir - 1, iz]
            - p_current[ir - 2, iz]
        )
        / 12
        / dr**2
    )
    ddr = (
        (
            -p_current[ir + 2, iz]
            + 8 * p_current[ir + 1, iz]
            - 8 * p_current[ir - 1, iz]
            + p_current[ir - 2, iz]
        )
        / 12
        / dr
    )
    ddz2 = (
        (
            -p_current[ir, iz + 2]
            + 16 * p_current[ir, iz + 1]
            - 30 * p_current[ir, iz]
            + 16 * p_current[ir, iz - 1]
            - p_current[ir, iz - 2]
        )
        / 12
        / dz**2
    )
    r = abs(ir - 2 + 0.5) * dr
    return ddr2 + ddz2 + ddr / r


@cuda.jit
def calc_dphi_dpsi_kernel_2D(
    p: np.ndarray,
    model_data: tuple[int, int, float, float, float, float, float, float, float],
    phipsi: np.ndarray,
    sigma: np.ndarray,
):
    # Unpack data
    nr, nz, dr, dz, dt, c0, beta, rho0, diffusivity = model_data

    ir, iz = cuda.grid(2)

    if ir > 1 and ir < nr - 2 and iz > 1 and iz < nz - 2:
        dp_term_r = (p[0, ir + 1, iz] - p[0, ir - 1, iz]) / (2 * dr)
        dp_term_z = (p[0, ir, iz + 1] - p[0, ir, iz - 1]) / (2 * dz)

        # r-axis
        # dphir
        phipsi[4, ir, iz] = (
            -0.5
            * (
                sigma[ir - 1, iz] * phipsi[0, ir - 1, iz]
                + sigma[ir, iz] * phipsi[0, ir, iz]
            )
            - dp_term_r
        )
        # dpsir
        phipsi[6, ir, iz] = (
            -0.5
            * (
                sigma[ir - 1, iz] * phipsi[2, ir, iz]
                + sigma[ir, iz] * phipsi[2, ir + 1, iz]
            )
            - dp_term_r
        )

        # z-axis
        # dphiz
        phipsi[5, ir, iz] = (
            -0.5
            * (
                sigma[ir, iz - 1] * phipsi[1, ir, iz - 1]
                + sigma[ir, iz] * phipsi[1, ir, iz]
            )
            - dp_term_z
        )
        # dpsiz
        phipsi[7, ir, iz] = (
            -0.5
            * (
                sigma[ir, iz - 1] * phipsi[3, ir, iz]
                + sigma[ir, iz] * phipsi[3, ir, iz + 1]
            )
            - dp_term_z
        )
    # if iz == 0:
    #     for i in range(4, 8):
    #         phipsi[i, ir, 0] = phipsi[i, ir, 3]
    # if iz == 1:
    #     for i in range(4, 8):
    #         phipsi[i, ir, 1] = phipsi[i, ir, 2]
    if ir == 0 and iz > 1 and iz < nz - 2:
        for i in range(4, 8):
            phipsi[i, 0, iz] = phipsi[i, 3, iz]
    if ir == 1 and iz > 1 and iz < nz - 2:
        for i in range(4, 8):
            phipsi[i, 1, iz] = phipsi[i, 2, iz]


@cuda.jit
def update_phi_psi_kernel_2D(
    model_data: tuple[int, int, float, float, float, float, float, float, float],
    phipsi: np.ndarray,
):
    # Unpack data
    nr, nz, dr, dz, dt, c0, beta, rho0, diffusivity = model_data

    ir, iz = cuda.grid(2)

    if ir > 1 and ir < nr - 2 and iz > 1 and iz < nz - 2:
        phipsi[0, ir, iz] += phipsi[4, ir, iz] * dt
        phipsi[1, ir, iz] += phipsi[5, ir, iz] * dt
        phipsi[2, ir, iz] += phipsi[6, ir, iz] * dt
        phipsi[3, ir, iz] += phipsi[7, ir, iz] * dt

    # if iz == 0:
    #     for i in range(4):
    #         phipsi[i, ir, 0] = phipsi[i, ir, 3]
    # if iz == 1:
    #     for i in range(4):
    #         phipsi[i, ir, 1] = phipsi[i, ir, 2]

    if ir == 0 and iz > 1 and iz < nz - 2:
        for i in range(4):
            phipsi[i, 0, iz] = phipsi[i, 3, iz]
    if ir == 1 and iz > 1 and iz < nz - 2:
        for i in range(4):
            phipsi[i, 1, iz] = phipsi[i, 2, iz]


@cuda.jit
def exchange_p_kernel_2D(
    model_data: tuple[int, int, float, float, float, float, float, float, float],
    p: np.ndarray,
):
    nr, nz, dr, dz, dt, c0, beta, rho0, diffusivity = model_data

    ir, iz = cuda.grid(2)

    # if ir > 1 and ir < nr - 2 and iz > 1 and iz < nz - 2:
    if ir >= 0 and ir < nr and iz >= 0 and iz < nz:
        p[4, ir, iz] = p[3, ir, iz]
        p[3, ir, iz] = p[2, ir, iz]
        p[2, ir, iz] = p[1, ir, iz]
        p[1, ir, iz] = p[0, ir, iz]
        p[0, ir, iz] = 0.0
