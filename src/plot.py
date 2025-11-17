import matplotlib.pyplot as plt
import numpy as np

from .contrast import weber_fechner_expfit, weber_fechner_fit, weber_fechner_samples
from .spaces.hsl import HSLstd
from .spaces.hsv import HSVstd
from .spaces.oklab import OKLAB
from .spaces.rgb import RGB


def weber_fechner_plot(rgb_base: RGB, rgb_ref: RGB, wfc_s: float, nb_samples: int = 32):
    plt.figure(figsize=(16, 10))
    plt.scatter([], [], color="black", label="Samples")
    plt.hlines(wfc_s, 0.0, 1.0, colors="black", label="Target")

    hslstd_x, hslstd_y = weber_fechner_samples(rgb_base, rgb_ref, nb_samples, RGB.to_hslstd, HSLstd.to_rgb, 2)
    hslstd_a, hslstd_b, hslstd_mse = weber_fechner_fit(rgb_base, rgb_ref, nb_samples, RGB.to_hslstd, HSLstd.to_rgb, 2)
    x_curve = np.linspace(min(hslstd_x), max(hslstd_x), nb_samples)
    y_curve = hslstd_a * x_curve + hslstd_b
    plt.plot(
        x_curve,
        y_curve,
        color="blue",
        label=f"WFC HSLstd on L (MSE={hslstd_mse:.2f}): y = {hslstd_a:.2f}x + {hslstd_b:.2f}",
        linewidth=1,
    )
    plt.scatter(hslstd_x, hslstd_y, color="blue")

    hsvstd_x, hsvstd_y = weber_fechner_samples(rgb_base, rgb_ref, nb_samples, RGB.to_hsvstd, HSVstd.to_rgb, 2)
    hsvstd_a, hsvstd_b, hsvstd_mse = weber_fechner_fit(rgb_base, rgb_ref, nb_samples, RGB.to_hsvstd, HSVstd.to_rgb, 2)
    x_curve = np.linspace(min(hsvstd_x), max(hsvstd_x), nb_samples)
    y_curve = hsvstd_a * x_curve + hsvstd_b
    plt.plot(
        x_curve,
        y_curve,
        color="green",
        label=f"WFC HSVstd on V (MSE={hsvstd_mse:.2f}): y = {hsvstd_a:.2f}x + {hsvstd_b:.2f}",
        linewidth=1,
    )
    plt.scatter(hsvstd_x, hsvstd_y, color="green")

    oklab_x, oklab_y = weber_fechner_samples(rgb_base, rgb_ref, nb_samples, RGB.to_oklab, OKLAB.to_rgb, 0)
    oklab_a, oklab_b, oklab_mse = weber_fechner_expfit(
        rgb_base, rgb_ref, nb_samples, RGB.to_oklab, OKLAB.to_rgb, 0, normalize=True
    )
    x_curve = np.linspace(min(oklab_x), max(oklab_x), nb_samples)
    y_curve = oklab_b * np.exp(oklab_a * x_curve) - 1
    plt.plot(
        x_curve,
        y_curve,
        color="red",
        label=f"WFC OkLab on L (MSE={oklab_mse:.2f}): y = {oklab_a:.2f}e({oklab_b:.2f}x)",
        linewidth=1,
    )
    oklab_wa, oklab_wb, oklab_wmse = weber_fechner_expfit(
        rgb_base, rgb_ref, nb_samples, RGB.to_oklab, OKLAB.to_rgb, 0, normalize=True, weighted_least_squares=True
    )
    y_curve = oklab_wb * np.exp(oklab_wa * x_curve) - 1
    plt.plot(
        x_curve,
        y_curve,
        color="darkred",
        label=f"WFC OkLab on L (MSE={oklab_wmse:.2f}) with weighted least squares: y = \
            {oklab_wa:.2f}e({oklab_wb:.2f}x)",
        linewidth=1,
    )
    plt.scatter(oklab_x, oklab_y, color="red")

    plt.title("Numpy polyfit based on WFC in different spaces and dimensions")
    plt.xlabel("Modified dimension ([0; 1])")
    plt.ylabel("Weber Feshner Contrast")
    plt.legend()
    plt.grid(True)
    plt.show()
