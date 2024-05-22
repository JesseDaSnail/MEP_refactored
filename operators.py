import numpy as np
from scipy.signal import butter, filtfilt


def ddx(p, dx, axis=0):
    """
    Calculate the first spatial derivative of a 3D array along a given axis.
    """

    dp = np.zeros(p.shape)

    if axis == 0:
        pass
    elif axis == 1:
        p = np.transpose(p, (1, 0, 2))
        dp = np.transpose(dp, (1, 0, 2))
    elif axis == 2:
        p = np.transpose(p, (2, 1, 0))
        dp = np.transpose(dp, (2, 1, 0))
    else:
        raise ValueError("Incorrect axis.")

    N = dp.shape[0]
    dp[2 : N - 2, :] = (
        -1.0 / 12 * p[4:N, :]
        + 2.0 / 3 * p[3 : N - 1, :]
        - 2.0 / 3 * p[1 : N - 3, :]
        + 1.0 / 12 * p[: N - 4, :]
    )

    if axis == 1:
        dp = np.transpose(dp, (1, 0, 2))
    elif axis == 2:
        dp = np.transpose(dp, (2, 1, 0))

    return dp / dx


def ddx2(p, dx, axis=0):
    """
    Calculate the second spatial derivative of a 3D array along a given axis.
    """

    d2p = np.zeros(p.shape)

    if axis == 0:
        pass
    elif axis == 1:
        p = np.transpose(p, (1, 0, 2))
        d2p = np.transpose(d2p, (1, 0, 2))
    elif axis == 2:
        p = np.transpose(p, (2, 1, 0))
        d2p = np.transpose(d2p, (2, 1, 0))
    else:
        raise ValueError("Incorrect axis.")

    N = d2p.shape[0]
    d2p[2 : N - 2, :] = (
        -1.0 / 12 * p[4:N, :]
        + 4.0 / 3 * p[3 : N - 1, :]
        - 5.0 / 2 * p[2 : N - 2, :]
        + 4.0 / 3 * p[1 : N - 3, :]
        - 1.0 / 12 * p[: N - 4, :]
    )

    if axis == 1:
        d2p = np.transpose(d2p, (1, 0, 2))
    elif axis == 2:
        d2p = np.transpose(d2p, (2, 1, 0))

    return d2p / dx**2


def laplace(p, dx):
    """
    Calculate the Laplacian of a 3D array.
    """

    pxx = ddx2(p, dx, axis=0)
    if p.shape[1] > 1:
        pyy = ddx2(p, dx, axis=1)
    else:
        pyy = 0
    pzz = ddx2(p, dx, axis=2)
    return pxx + pyy + pzz


def ddt(p, dt):
    """
    Calculate the first temporal derivative of a 3D array.
    """

    return (p[0] - p[1]) / dt


def ddt2(p, dt):
    """
    Calculate the second temporal derivative of a 3D array.
    """

    if p.shape[0] < 3:
        return np.zeros(p[0].shape)
    return (p[2] - 2 * p[1] + p[0]) / dt**2


def ddt3(p, dt):
    """
    Calculate the third temporal derivative of a 3D array.
    """

    if p.shape[0] < 6:
        return np.zeros(p[0].shape)
    return (6 * p[0] - 23 * p[1] + 34 * p[2] - 24 * p[3] + 8 * p[4] - p[5]) / (
        2 * dt
    ) ** 3


def butter_lowpass_filter(p_response, cutoff, fs, order):
    # Get the filter coefficients
    b, a = butter(order, cutoff, fs=fs, btype="low")
    # Apply filter
    y = filtfilt(b, a, p_response, axis=0)
    return y
