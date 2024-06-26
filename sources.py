import numpy as np
from scipy.signal.windows import tukey, nuttall
import matplotlib.pyplot as plt
from model import Model


class Source:
    def __init__(
        self,
        ir: int,
        iz: int,
        iwidth: int,
        nt: int,
        dt: float,
        dr: float,
        dz: float,
        ist: int,
        max_freq: float,
        scaling_factor: float,
    ) -> None:
        """
        Initialize a source.

        Parameters:
        - ir, iz: Index coordinates of the source.
        - iwidth: Source width in number of grid points.
        - nt: Number of time steps.
        - dt: Time step size.
        - dr, dz: Grid spacing.
        - ist: Shifting of the source time function.
        - max_freq: Maximum frequency of the source.
        - scaling_factor: Scaling factor of the source.
        """

        if nt <= 0 or dt <= 0:
            raise ValueError("nt and dt must be positive values.")
        self.ir = ir
        self.iz = iz
        self.iwidth = iwidth
        self.nt = nt
        self.dt = dt
        self.dr = dr
        self.dz = dz
        self.ist = ist
        self.max_freq = max_freq
        self.scaling_factor = scaling_factor
        self.profile = self.generate_profile()
        self.r_focus = None
        self.y_focus = None
        self.z_focus = None
        self.phase_profile = np.zeros(self.profile.shape, dtype=np.int16)
        self.response = np.zeros(nt)

    def generate_profile(self, alpha: float = 0.5):
        """
        Generates the spatial source profile.

        Args:
            alpha (float, optional): The alpha value used in the Tukey window function. Defaults to 0.5.

        Returns:
            numpy.ndarray: The generated source profile.
        """
        profile_r = tukey(self.iwidth * 2, alpha)[self.iwidth :]
        return profile_r

    def set_focus(self, r: float, z: float, c0: float):
        """
        Set the focus of the source at the given coordinates (r, z) and calculate the phase profile based on the speed of sound (c0).

        Parameters:
            r (float): The r-coordinate of the focus.
            z (float): The z-coordinate of the focus.
            c0 (float): The speed of sound.

        Returns:
            None
        """
        self.r_focus = r
        self.z_focus = z

        rs = np.linspace(self.ir, self.ir + self.iwidth, self.iwidth) * self.dr

        R = np.sqrt((rs - r) ** 2 + (self.iz * self.dr - z) ** 2)
        time_phase = R / c0
        time_phase_index = np.round(time_phase / self.dt).astype(np.int16)
        self.phase_profile = time_phase_index.T

    def plot_info(self, model: Model = None):
        """
        Plot the information of the source.

        This function creates a subplot mosaic with four subplots: 'profile', 'phase_profile', 'response', and 'domain'.
        The 'profile' subplot displays the spatial source profile.
        The 'phase_profile' subplot displays the time shift index of the phase profile due to focus.
        The 'response' subplot displays the amplitude of the response.
        The 'domain' subplot displays the sigma values of the model, if provided.

        Parameters:
            model (Model, optional): The model object containing the sigma values. Defaults to None.

        Returns:
            None
        """
        fig, axes = plt.subplot_mosaic(
            [
                ["profile", "domain"],
                ["phase_profile", "domain"],
                ["response", "response"],
            ],
            figsize=(10, 10),
        )

        axes["profile"].plot(self.profile)
        axes["profile"].set_title("Profile")
        axes["profile"].set_ylabel("Amplitude multiplier")
        axes["profile"].set_xlabel("r index")

        axes["phase_profile"].plot(self.phase_profile)
        axes["phase_profile"].set_title("Phase Profile")
        axes["phase_profile"].set_ylabel("Time shift index")
        axes["phase_profile"].set_xlabel("r index")

        axes["response"].plot(np.arange(0, self.nt) * self.dt, self.response)
        axes["response"].set_title("Response")
        axes["response"].set_ylabel("Amplitude")
        axes["response"].set_xlabel("Time (s)")

        if model is not None:
            axes["domain"].imshow(
                model.sigma.T,
                extent=[
                    -2 * model.dr,
                    (model.nrd + model.pml_width) * model.dr,
                    (model.nzd + model.pml_width) * model.dz,
                    -model.pml_width * model.dz,
                ],
            )
            axes["domain"].set_title("Sigma")
            axes["domain"].set_xlabel("r (m)")
            axes["domain"].set_ylabel("z (m)")

            source_r = (np.arange(self.ir, self.ir + self.iwidth)) * model.dr
            source_z = (np.ones(self.iwidth) * self.iz - model.pml_width) * model.dz
            axes["domain"].scatter(
                source_r,
                source_z,
                s=5,
                color="r",
            )

        if self.r_focus is not None:
            axes["domain"].scatter(self.r_focus, self.z_focus, s=5, color="lime")

        plt.tight_layout()
        plt.show()

    def __str__(self) -> str:
        return (
            f"Source:\n"
            f"  ir={self.ir}\n"
            f"  iz={self.iz}\n"
            f"  iwidth={self.iwidth} / {self.iwidth * self.dr} m\n"
            f"  pulse_length={self.pulse_length}\n"
            f"  max_freq={self.max_freq}\n"
        )


class CosineSource(Source):
    def __init__(
        self,
        ir: int,
        iz: int,
        iwidth: int,
        amplitude: float,
        frequency: float,
        nt: int,
        dt: float,
        dr: float,
        dz: float,
        ist: int,
        pulse_length: int,
    ) -> None:
        """
        Initialize a cosine source.

        Parameters:
        - pulse_width: Width of the pulse in the time domain.
        """
        scaling_factor = 1 / (0.18178913540596034 * pulse_length)
        super().__init__(ir, iz, iwidth, nt, dt, dr, dz, ist, frequency, scaling_factor)
        self.amplitude = amplitude
        self.frequency = frequency
        self.pulse_length = pulse_length
        self.response = self.generate_response()

    def generate_response(self):
        time_range = np.arange(0, self.nt) * self.dt
        src = np.cos(2 * np.pi * self.frequency * (time_range - self.ist * self.dt))
        window_term = nuttall(self.pulse_length)
        window_term = np.pad(
            window_term, (self.ist, self.nt - self.ist - len(window_term))
        )
        src *= window_term
        width = self.iwidth * self.dr
        src *= self.max_freq * self.amplitude / width
        return src


class SimpleSource(Source):
    def __init__(
        self,
        ir: int,
        iz: int,
        iwidth: int,
        amplitude: float,
        frequency: float,
        nt: int,
        dt: float,
        dr: float,
        dz: float,
        ist: int,
    ) -> None:
        raise NotImplementedError("SimpleSource is not implemented correctly yet.")
        super().__init__(ir, iz, iwidth, nt, dt, dr, dz, ist, frequency)
        self.amplitude = amplitude
        self.frequency = frequency
        self.response = self.generate_response()

    def generate_response(self):
        # Source time function Gaussian, nt + 1 as we loose the last one by diff
        src = np.empty(self.nt + 1)
        for it in range(self.nt):
            src[it] = np.exp(
                -((2 * np.pi * self.frequency) ** 2) * ((it - self.ist) * self.dt) ** 2
            )
        # Take the first derivative
        src = np.diff(src) / self.dt
        src[self.nt - 1] = 0
        src /= src.max()
        width = self.iwidth * self.dr
        src *= self.max_freq * self.amplitude / width
        return src


class RickerSource(Source):
    def __init__(
        self,
        ir: int,
        iz: int,
        iwidth: int,
        amplitude: float,
        frequency: float,
        nt: int,
        dt: float,
        dr: float,
        dz: float,
        ist: int,
    ) -> None:
        raise NotImplementedError("RickerSource is not implemented correctly yet.")
        super().__init__(ir, iz, iwidth, nt, dt, dr, dz, ist, frequency)
        self.amplitude = amplitude
        self.frequency = frequency
        self.response = self.generate_response()

    def generate_response(self):
        # Source time function Gaussian, nt + 1 as we loose the last one by diff
        src = np.empty(self.nt + 2)
        for it in range(self.nt):
            src[it] = np.exp(
                -((2 * np.pi * self.frequency) ** 2) * ((it - self.ist) * self.dt) ** 2
            )
        # Take the first derivative
        src = np.diff(np.diff(src)) / self.dt
        src[self.nt - 1] = 0
        src /= src.max()
        width = self.iwidth * self.dr
        src *= self.max_freq * self.amplitude / width
        return src


class OscillatorSource(Source):
    def __init__(
        self,
        ir: int,
        iz: int,
        iwidth: int,
        amplitude: float,
        frequency: float,
        nt: int,
        dt: float,
        dr: float,
        dz: float,
        ist: int,
    ) -> None:
        raise NotImplementedError("OscillatorSource is not implemented correctly yet.")
        super().__init__(ir, iz, iwidth, nt, dt, dr, dz, ist, frequency)
        self.amplitude = amplitude
        self.frequency = frequency
        self.response = self.generate_response()

    def generate_response(self):
        time_range = np.arange(0, self.nt) * self.dt
        src = np.sin(2 * np.pi * self.frequency * (time_range - self.ist))
        width = self.iwidth * self.dr
        src *= self.max_freq * self.amplitude / width
        return src


class ParametricSource(Source):
    def __init__(
        self,
        ir: int,
        iz: int,
        iwidth: int,
        amplitude: float,
        frequency1: float,
        frequency2: float,
        nt: int,
        dt: float,
        dr: float,
        dz: float,
        ist: int,
        pulse_length: int,
    ) -> None:
        """
        Initialize a cosine source.

        Parameters:
        - pulse_length: Length of the pulse in the time domain in time steps.
        """
        scaling_factor = 1 / (0.18178913540596034 * pulse_length)
        super().__init__(
            ir,
            iz,
            iwidth,
            nt,
            dt,
            dr,
            dz,
            ist,
            max(frequency1, frequency2),
            scaling_factor,
        )
        self.amplitude = amplitude
        self.frequency1 = frequency1
        self.frequency2 = frequency2
        self.pulse_length = pulse_length
        self.response = self.generate_response()

    def generate_response(self):
        time_range = np.arange(0, self.nt) * self.dt
        cos_term = np.cos(
            2 * np.pi * self.frequency1 * (time_range - self.ist * self.dt)
        ) + np.cos(2 * np.pi * self.frequency2 * (time_range - self.ist * self.dt))

        window_term = nuttall(self.pulse_length)
        window_term = np.pad(
            window_term, (self.ist, self.nt - self.ist - len(window_term))
        )
        src = cos_term * window_term
        width = self.iwidth * self.dr
        src *= self.max_freq * self.amplitude / width
        return src
