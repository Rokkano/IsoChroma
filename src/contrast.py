from typing import Callable, Tuple

import numpy as np

from src.spaces.abstract import ColorSpace
from src.spaces.hsl import HSL
from src.spaces.rgb import RGB

GAMMA_CORRECTION: float = 2.4


def weber_fechner_contrast(rgb_fg: RGB, rgb_bg: RGB) -> float:
    def luminance(rgb: RGB) -> float:
        def Clin(C: float) -> float:
            return C / 12.92 if C >= 0.04045 else ((C + 0.055) / (1 + 0.055)) ** GAMMA_CORRECTION

        R, G, B = rgb.values
        return 0.2126 * Clin(R) + 0.7152 * Clin(G) + 0.0722 * Clin(B)

    Lzone = luminance(rgb_fg)
    Lfond = luminance(rgb_bg)
    return (Lzone - Lfond) / Lfond


def weber_fechner_samples(
    rgb_base: RGB,
    rgb_ref: RGB,
    nb_samples: int = 32,
    color_space_conv: Callable[[RGB], ColorSpace] = lambda rgb: RGB.to_hsl(rgb),
    color_space_conv_inv: Callable[[ColorSpace], RGB] = lambda hsl: HSL.to_rgb(hsl),
    color_space_dim: int = 2,
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    # Splitting target (default to HSL) into nb_samples values of target dim and fitting on the inversed WBC
    x, y = [], []
    for index in range(0, nb_samples):
        target = color_space_conv(rgb_base)
        target[color_space_dim] = index / nb_samples
        x.append(index / nb_samples)
        y.append(weber_fechner_contrast(color_space_conv_inv(target), rgb_ref))

    x = np.array(x)
    y = np.array(y)
    return x, y


def weber_fechner_fit(
    rgb_base: RGB,
    rgb_ref: RGB,
    nb_samples: int = 32,
    color_space_conv: Callable[[RGB], ColorSpace] = lambda rgb: RGB.to_hsl(rgb),
    color_space_conv_inv: Callable[[ColorSpace], RGB] = lambda hsl: HSL.to_rgb(hsl),
    color_space_dim: int = 2,
) -> Tuple[float, float, float]:
    x, y = weber_fechner_samples(rgb_base, rgb_ref, nb_samples, color_space_conv, color_space_conv_inv, color_space_dim)

    # Fitting on linear ax + b
    a, b = np.polyfit(x, y, 1)
    mse = 1 / nb_samples * np.sum(np.pow(y - (a * x + b), 2))
    return a, b, mse


def weber_fechner_logfit(
    rgb_base: RGB,
    rgb_ref: RGB,
    nb_samples: int = 32,
    color_space_conv: Callable[[RGB], ColorSpace] = lambda rgb: RGB.to_hsl(rgb),
    color_space_conv_inv: Callable[[ColorSpace], RGB] = lambda hsl: HSL.to_rgb(hsl),
    color_space_dim: int = 2,
    normalize: bool = True,
) -> Tuple[float, float]:
    x, y = weber_fechner_samples(rgb_base, rgb_ref, nb_samples, color_space_conv, color_space_conv_inv, color_space_dim)

    # Fitting on logarithmic y = a + b * log(x)
    # Normalizing on x is only adding an epsilon since the first value can be 0.
    a, b = np.polyfit(x=np.log(x + 1e-5) if normalize else x, y=y, deg=1)

    mse = 1 / nb_samples * np.sum(np.pow(y - (a * np.log(x) + b), 2))
    return a, b, mse


def weber_fechner_expfit(
    rgb_base: RGB,
    rgb_ref: RGB,
    nb_samples: int = 32,
    color_space_conv: Callable[[RGB], ColorSpace] = lambda rgb: RGB.to_hsl(rgb),
    color_space_conv_inv: Callable[[ColorSpace], RGB] = lambda hsl: HSL.to_rgb(hsl),
    color_space_dim: int = 2,
    normalize: bool = True,
    weighted_least_squares: bool = False,
) -> Tuple[float, float]:
    x, y = weber_fechner_samples(rgb_base, rgb_ref, nb_samples, color_space_conv, color_space_conv_inv, color_space_dim)

    # Fitting on exponential y = ae^(bx)
    # Normalizing on y is adding 1 since the values for oklab are within [-1; inf].
    a, b = np.polyfit(
        x=x, y=np.log(y + 1) if normalize else np.log(y), deg=1, w=np.sqrt(y + 1) if weighted_least_squares else None
    )
    b = np.exp(b)
    mse = 1 / nb_samples * np.sum(np.pow(y - (b * np.exp(a * x) - (1 if normalize else 0)), 2))
    return a, b, mse
