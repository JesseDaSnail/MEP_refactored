import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import cupy as cp
from model import Model
from sources import Source
from simulation import SimulationResult


class SimulationResultAnalyser:
    def __init__(self, simulation_result: SimulationResult):
        self.p = simulation_result.p
        self.model = simulation_result.model
        self.sources = simulation_result.sources
        self.subsample_space = simulation_result.subsample_space
        self.subsample_time = simulation_result.subsample_time

    def plot_slices_time(self, window_size: int = None, vlimit: float = None):
        """
        Plots slices of the pressure field along the time axis at different depths.

        Parameters:
            window_size (int, optional): The size of the window to plot around the maximum pressure value. Defaults to None.
            vlimit (float, optional): The upper limit for the y-axis of the plot. Defaults to None.

        Returns:
            None
        """

        # Unpack variables
        p = self.p
        dr = self.model.dr * self.subsample_space
        dt = self.model.dt * self.subsample_time

        fig, ax = plt.subplots(2, 3, figsize=(10, 10))

        t_values = np.arange(0, p.shape[0] * dt, dt)
        z_values = np.linspace(
            p.shape[2] / 6, p.shape[2] - p.shape[2] / 6, 6, dtype=np.int64
        )

        if window_size is None:
            window_size = p.shape[2] / 10
        if vlimit is None:
            vlimit = p.max()
        itr = 0
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                to_plot = z_values[itr]

                max_index = np.argmax(p[:, 2, to_plot])
                idx1 = int(max_index - window_size)
                idx2 = int(max_index + window_size)

                ax[i, j].plot(t_values[idx1:idx2], p[idx1:idx2, 2, to_plot])

                ax[i, j].set_title(f"Depth: {to_plot*dr:.2f} m")
                ax[i, j].set_xlabel("t (s)")
                ax[i, j].set_ylabel("p (Pa)")
                ax[i, j].set_ylim(-vlimit, vlimit)
                itr += 1

        plt.tight_layout()
        plt.plot()

    def plot_snapshots(
        self, vlimit: float = None, mirror: bool = False, title_type: str = "iteration"
    ):
        """
        Plots snapshots of the result.

        Args:
            vlimit (float, optional): The limits for the colorbar of the plot. Defaults to None.
            mirror (bool, optional): Whether to mirror the pressure field over the z-axis. Defaults to False.
            title_type (str, optional): The type of title to display. Must be either "iteration" or "time". Defaults to "iteration".

        Raises:
            ValueError: If title_type is not either "iteration" or "time".

        Returns:
            None
        """
        if title_type not in ["iteration", "time"]:
            raise ValueError("title_type must be either 'iteration' or 'time'")

        # Unpack variables
        p = self.p
        subsample_time = self.subsample_time
        dr = self.model.dr * self.subsample_space
        dt = self.model.dt * self.subsample_time

        # Plot snapshots of result
        fig, ax = plt.subplots(2, 3, figsize=(10, 10))

        if vlimit is None:
            vlimit = p.max()

        iterations = np.linspace(0, p.shape[0] - 1, 6, dtype=np.int64)
        itr = 0
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                to_plot = iterations[itr]

                image = p[to_plot, :, :].T
                if mirror:
                    image = np.concatenate((np.flip(image, axis=1), image), axis=1)

                # Plot snapshot of pressure field
                im = ax[i, j].imshow(
                    image,
                    extent=[
                        -p.shape[1] * dr if mirror else 0,
                        p.shape[1] * dr,
                        p.shape[2] * dr,
                        0,
                    ],
                    origin="upper",
                    vmin=-vlimit,
                    vmax=vlimit,
                )

                # fig.colorbar(im, shrink=0.3)
                fig.colorbar(im, shrink=0.8)
                if title_type == "iteration":
                    ax[i, j].set_title(f"Iteration: {to_plot*subsample_time}")
                elif title_type == "time":
                    ax[i, j].set_title(f"Time: {to_plot*dt:.2f} s")
                ax[i, j].set_xlabel("x (m)")
                ax[i, j].set_ylabel("z (m)")
                itr += 1

        # plt.subplots_adjust(wspace=0.4,
        #                     hspace=-0.8)
        plt.tight_layout()
        plt.show()

    def generate_gif(
        self,
        filename: str,
        fps: int = 5,
        frame_interval: int = 10,
        vlimit: float = None,
        mirror: bool = False,
    ):
        """
        Generates a GIF animation from the pressure field data.

        Parameters:
            filename (str): The name of the file to save the GIF animation.
            fps (int, optional): The frames per second of the animation. Defaults to 5.
            frame_interval (int, optional): The interval between frames in the pressure field data. Defaults to 10.
            vlimit (float, optional): The upper limit for the colorbar of the animation. Defaults to None.

        Returns:
            None
        """
        # Create a figure and axis
        fig, ax = plt.subplots()

        if vlimit is None:
            vlimit = self.p.max()

        p = self.p
        dr = self.model.dr * self.subsample_space
        dz = self.model.dz * self.subsample_space
        dt = self.model.dt * self.subsample_time

        # Initialize the image plot
        image = ax.imshow(
            self.p[0, :, :].T,
            extent=[
                -p.shape[1] * dr if mirror else 0,
                p.shape[1] * dr,
                p.shape[2] * dz,
                0,
            ],
            origin="upper",
            vmin=-vlimit,
            vmax=vlimit,
        )

        plt.xlabel("r (m)")
        plt.ylabel("z (m)")

        # Update function for the animation
        def update(frame):
            index = frame * frame_interval
            current_frame = self.p[index, :, :].T
            if mirror:
                current_frame = np.concatenate(
                    (np.flip(current_frame, axis=1), current_frame), axis=1
                )
            image.set_array(current_frame)
            return (image,)

        num_frames = p.shape[0] // frame_interval

        # Create the animation
        ani = FuncAnimation(
            fig,
            update,
            frames=tqdm(range(num_frames)),
            interval=1e3 / fps,
        )

        # Save the animation as a GIF
        ani.save(filename, writer="pillow", fps=fps)  # Adjust FPS as needed
        plt.close()

    def harmonic_progression(
        self,
        num_harmonics: int = 5,
        r_start: int = None,
        z_start: int = None,
        angle: float = 0,
        include_subharmonic: bool = False,
        subharmonic_freq: float = None,
        scaling_factor: float = None,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the harmonic progression of the pressure response along a line.
        This line is defined by a starting point and an angle in the rz-plane.

        Parameters
        ----------
        num_harmonics : int, optional
            Number of harmonics to calculate, by default 10
        r_start : int, optional
            r-coordinate of the starting point of the line, by default None
        z_start : int, optional
            z-coordinate of the starting point of the line, by default None
        angle : float, optional
            Angle of the line with respect to the z-axis, by default 0
        include_subharmonic : bool, optional
            Flag indicating whether to include the subharmonic, by default False
        subharmonic_freq : float, optional
            Frequency of the subharmonic, by default None
        scaling_factor : float, optional
            Scaling factor for the harmonic amplitude, by default None
        normalize : bool, optional
            Flag indicating whether to normalize the harmonic progression, by default True

        Returns
        -------
        distances : np.ndarray
            Distances along the line
        harmonic_progression_array : np.ndarray
            Array containing the harmonic progression for each distance
        harmonics : np.ndarray
            Array containing the harmonic frequencies

        """
        if scaling_factor is None:
            raise ValueError(
                "Please provide a scaling factor for the harmonic progression."
            )
        if num_harmonics * self.sources[0].max_freq > 1 / (
            2 * self.model.dt * self.subsample_time
        ):
            num_harmonics = int(
                1 / (2 * self.model.dt * self.subsample_time) / self.sources[0].max_freq
            )
            print(
                f"Highest harmonic not discernible at dt = {self.model.dt * self.subsample_time}."
                + f"Setting num_harmonics to {num_harmonics}."
            )

        # Unpack variables
        p = self.p
        dt = self.model.dt * self.subsample_time
        dr = self.model.dr * self.subsample_space

        # Select pressure values and indices along the line
        p_selected, x_indices, z_indices = self.select_line(
            r_start=r_start,
            z_start=z_start,
            angle=angle,
        )

        distances = (
            np.sqrt((x_indices - r_start) ** 2 + (z_indices - z_start) ** 2) * dr
        )

        # Define frequencies of the harmonics
        harmonics = self.sources[0].max_freq * np.arange(1, num_harmonics + 1)
        if include_subharmonic:
            if subharmonic_freq is None:
                subharmonic_freq = self.sources[0].max_freq / 2
            harmonics = np.insert(harmonics, 0, subharmonic_freq)
            num_harmonics += 1

        # Calculate the frequency array
        freq_array = np.fft.fftfreq(p.shape[0], dt)[: p.shape[0] // 2]

        # Find the indices of the frequencies closest to the harmonic frequencies
        harmonic_indices = np.zeros(num_harmonics, dtype=np.int64)
        for i, value in enumerate(harmonics):
            idx = (np.abs(freq_array - value)).argmin()
            harmonic_indices[i] = idx

        # Calculate the harmonic progression for each harmonic over distance
        harmonic_progression_array = np.zeros((num_harmonics, distances.size))
        for i in range(distances.size):
            p_response = p_selected[:, i]
            freq_array, dB_response = get_dB_response(
                p_response,
                dt,
                scaling_factor=scaling_factor,
                normalize=normalize,
            )
            harmonic_progression_array[:, i] = dB_response[harmonic_indices]

        return distances, harmonic_progression_array, harmonics

    def select_line(
        self,
        r_start: int = None,
        z_start: int = None,
        angle: float = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Select values from the 3D array along a line. This line is defined by a
        starting point and an angle in the rz-plane.

        Parameters
        ----------
        r_start : int, optional
            r-coordinate index of the starting point of the line, by default None
        z_start : int, optional
            z-coordinate index of the starting point of the line, by default None
        angle : float, optional
            Angle of the line with respect to the z-axis, by default 0

        Returns
        -------
        selected_values : np.ndarray
            Values selected along the line
        r_indices : np.ndarray
            Indices of the selected values along the r-axis
        z_indices : np.ndarray
            Indices of the selected values along the z-axis
        """
        p = self.p

        if r_start is None:
            r_start = 0
        if z_start is None:
            z_start = 0

        t = np.arange(0, np.sqrt(p.shape[1] ** 2 + p.shape[2] ** 2))
        r = r_start + np.sin(angle) * t
        z = z_start + np.cos(angle) * t

        # Round to the nearest integer to get indices
        r_indices = np.rint(r).astype(int)
        z_indices = np.rint(z).astype(int)

        # Mask indices outside of array bounds
        valid_indices = (
            (r_indices >= 0)
            & (r_indices < p.shape[1])
            & (z_indices >= 0)
            & (z_indices < p.shape[2])
        )

        # Use only valid indices
        r_indices = r_indices[valid_indices]
        z_indices = z_indices[valid_indices]

        # Extract values from the 3D array using the calculated indices
        selected_values = p[:, r_indices, z_indices]

        return selected_values, r_indices, z_indices

    def get_lobes(self, frequency: float) -> np.ndarray:
        """
        Computes the fourier coefficient of the pressure response over the whole
        domain for a given frequency.

        Parameters
        ----------
        frequency : float
            Frequency of the fourier coefficient.

        Returns
        -------
        lobe_array : np.ndarray
            Array containing the fourier coefficient over the whole domain.

        """
        p = self.p
        dt = self.model.dt * self.subsample_time

        lobe_array = np.zeros((p.shape[1], p.shape[2]))
        correlation_function = np.exp(
            -1j * 2 * np.pi * frequency * np.arange(p.shape[0]) * dt
        )
        for r in tqdm(range(p.shape[1])):
            for z in range(p.shape[2]):
                lobe_array[r, z] = np.abs(
                    np.correlate(
                        self.p[:, r, z],
                        correlation_function,
                    )
                )

        return lobe_array * 2 / p.shape[0]

    def get_lobes_all(
        self, scaling_factor: float, subsample_space: int = 1
    ) -> np.ndarray:
        """
        Computes the Fourier coefficients of the pressure response over the whole domain for each frequency.

        Parameters:
            scaling_factor (float): The scaling factor for the Fourier coefficients.
            subsample_space (int, optional): The spatial subsampling factor for the pressure field. Defaults to 1.

        Returns:
            numpy.ndarray: An array of frequencies.
            numpy.ndarray: An array of Fourier coefficients.

        """
        mem_max = 1e9
        if self.p.nbytes > mem_max:
            factor = self.p.nbytes / mem_max
            if subsample_space < factor:
                subsample_space = int(np.ceil(factor))
                print(
                    f"Array too large for GPU. Setting subsample to {subsample_space}."
                )

        p = self.p
        dt = self.model.dt * self.subsample_time

        frequencies = np.fft.fftfreq(p.shape[0], dt)

        p_device = cp.asarray(p[:, ::subsample_space, ::subsample_space])
        lobe_array_device = cp.fft.fft(p_device, axis=0)
        lobe_array_device = cp.abs(lobe_array_device)
        lobe_array_device *= scaling_factor
        lobe_array = cp.asnumpy(lobe_array_device)

        del p_device
        del lobe_array_device
        cp.get_default_memory_pool().free_all_blocks()

        return frequencies[: frequencies.size // 2], lobe_array[: frequencies.size // 2]

    def get_lobes_all_split(
        self, scaling_factor: float, subsample_space: int = 1
    ) -> np.ndarray:
        """
        Same as `get_lobes_all` but splits the domain to make it more memory efficient.
        Computes the Fourier coefficients of the pressure response over the whole domain for each frequency.
        The pressure field is split into subdomains and the Fourier coefficients are computed for each subdomain.

        Parameters:
            scaling_factor (float): The scaling factor for the Fourier coefficients.
            subsample (int, optional): The spatial subsampling factor for the pressure field. Defaults to 1.

        Returns:
            numpy.ndarray: An array of frequencies.
            numpy.ndarray: An array of Fourier coefficients.

        """
        mem_max = 1e9
        if self.p.nbytes > mem_max:
            factor = self.p.nbytes / mem_max
            factor = int(np.ceil(factor))
            print(f"Array too large for GPU. Splitting domain.")

        p = self.p
        dt = self.model.dt * self.subsample_time

        p_subsampled = p[:, ::subsample_space, ::subsample_space]

        frequencies = np.fft.fftfreq(p.shape[0], dt)
        lobe_array = np.zeros(p_subsampled.shape)

        # Distribute and filter p
        for i in tqdm(range(factor)):
            p_device = cp.asarray(
                p_subsampled[
                    :,
                    :,
                    i
                    * p_subsampled.shape[2]
                    // factor : (i + 1)
                    * p_subsampled.shape[2]
                    // factor,
                ]
            )
            p_device = cp.abs(cp.fft.fft(p_device, axis=0)) * scaling_factor

            lobe_array[
                :,
                :,
                i
                * p_subsampled.shape[2]
                // factor : (i + 1)
                * p_subsampled.shape[2]
                // factor,
            ] = cp.asnumpy(p_device)

            del p_device
            cp.get_default_memory_pool().free_all_blocks()

        return frequencies[: frequencies.size // 2], lobe_array[: frequencies.size // 2]


def get_frequency_response(
    pressure_response: np.ndarray, dt: float
) -> tuple[np.ndarray]:
    """
    Calculates the frequency response of a pressure response by applying a
    Fourier transform.

    Parameters:
        pressure_response (ndarray): One-dimensional array containing the pressure response.
        dt (float): Time step between samples.

    Returns:
        freq_array (ndarray): Array of frequencies.
        frequency_response (ndarray): Array of Fourier coefficients.
    """
    if pressure_response.ndim != 1:
        raise ValueError("pressure_response must be one dimensional.")

    nt = pressure_response.size
    frequency_response = np.fft.fft(pressure_response)[: nt // 2]
    freq_array = np.fft.fftfreq(nt, dt)[: nt // 2]

    return freq_array, np.abs(frequency_response)


def get_dB_response(
    pressure_response: np.ndarray,
    dt: float,
    scaling_factor: float,
    normalize: bool = True,
) -> tuple[np.ndarray]:
    """
    Calculates the frequency response using 'get_frequency_response()' and scales it to dB.

    Parameters:
        pressure_response (ndarray): One-dimensional array containing the pressure response.
        dt (float): Time step between samples.
        scaling_factor (float): The scaling factor for the Fourier coefficients.
        normalize (bool, optional): Flag indicating whether to normalize the dB response.
            Defaults to True.

    Returns:
        freq_array (ndarray): Array of frequencies.
        dB_response (ndarray): Array of dB response values.
    """

    if pressure_response.ndim != 1:
        raise ValueError("pressure_response must be one dimensional.")

    freq_array, frequency_response = get_frequency_response(pressure_response, dt)
    # frequency_response *= (
    #     2 / pressure_response.size
    # )
    frequency_response *= scaling_factor
    reference_pressure = 1e-6
    epsilon = 1e-10
    dB_response = 20 * np.log10((frequency_response + epsilon) / reference_pressure)
    if normalize:
        dB_response -= dB_response.max()
    return freq_array, dB_response
