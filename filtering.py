import numpy as np
import cupy as cp
from cupyx.scipy.signal import butter, filtfilt
from scipy import signal


def lp_filter(p, dt, cutoff):
    mem_max = 1e9
    factor = 1
    if p.nbytes > mem_max:
        factor = p.nbytes / mem_max
        factor = int(np.ceil(factor))
        print(f"lp_filter: Array too large for GPU. Splitting domain.")

    freq_array = np.fft.fftfreq(p.shape[0], d=dt)

    # Distribute and filter p
    for i in range(factor):
        p_device = cp.asarray(
            p[
                :,
                :,
                i * p.shape[2] // factor : (i + 1) * p.shape[2] // factor,
            ]
        )
        p_device = cp.fft.fft(p_device, axis=0)
        p_device[np.abs(freq_array) > cutoff] = 0
        p_device = cp.fft.ifft(p_device, axis=0)
        p[
            :,
            :,
            i * p.shape[2] // factor : (i + 1) * p.shape[2] // factor,
        ] = cp.asnumpy(p_device).real

        del p_device
        cp.get_default_memory_pool().free_all_blocks()

    return p


def lp_filter2(p, dt, cutoff):
    mem_max = 0.3e9
    factor = 1
    if p.nbytes > mem_max:
        factor = p.nbytes / mem_max
        factor = int(np.ceil(factor))
        print(f"lp_filter: Array too large for GPU. Splitting domain.")

    freq_array = np.fft.fftfreq(p.shape[0], d=dt)
    delta_freq = np.maximum(np.abs(freq_array) - cutoff, 0)
    h = np.exp(-(1e-6 * delta_freq**2))
    h_device = cp.asarray(h)

    # Distribute and filter p
    for i in range(factor):
        p_device = cp.asarray(
            p[
                :,
                :,
                i * p.shape[2] // factor : (i + 1) * p.shape[2] // factor,
            ]
        )
        p_device = cp.fft.fft(p_device, axis=0)
        # p_device[np.abs(freq_array) > cutoff] = 0
        p_device = cp.fft.fftshift(cp.multiply(p_device, h_device[:, None, None]))
        p_device = cp.fft.ifftshift(p_device)
        p_device = cp.fft.ifft(p_device, axis=0)
        p[
            :,
            :,
            i * p.shape[2] // factor : (i + 1) * p.shape[2] // factor,
        ] = cp.asnumpy(p_device).real

        del p_device
        cp.get_default_memory_pool().free_all_blocks()

    return p


# TODO better filter
def lp_butter_filter(p, dt, cutoff, order=2):
    mem_max = 1e9
    if p.nbytes > mem_max:
        factor = p.nbytes / mem_max
        factor = int(np.ceil(factor))
        print(f"lp_filter: Array too large for GPU. Splitting domain.")

    # Get the filter coefficients
    b, a = butter(order, cutoff, fs=1 / dt, btype="low")

    # freq_array = np.fft.fftfreq(p.shape[0], d=dt)

    # Distribute and filter p
    for i in range(factor):
        # p_device = cp.asarray(
        #     p[
        #         :,
        #         :,
        #         i * p.shape[2] // factor : (i + 1) * p.shape[2] // factor,
        #     ]
        # )

        # Apply filter
        p_new = filtfilt(
            b,
            a,
            p[
                :,
                :,
                i * p.shape[2] // factor : (i + 1) * p.shape[2] // factor,
            ],
            axis=0,
        )

        p[
            :,
            :,
            i * p.shape[2] // factor : (i + 1) * p.shape[2] // factor,
        ] = p_new

        del p_new
        # cp.get_default_memory_pool().free_all_blocks()

    return p
