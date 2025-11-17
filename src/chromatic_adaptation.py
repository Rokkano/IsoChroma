from typing import Literal

import numpy as np
import pandas as pd

illuminant_space_names = Literal["A", "B", "C", "D50", "D55", "D65", "D75", "E", "F2", "F7", "F11"]

scaling_methods = Literal["XYZ", "Bradford", "VonKries"]
scaling_matrixes = {
    "XYZ": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    "Bradford": [
        [0.8951000, 0.2664000, -0.1614000],
        [-0.7502000, 1.7135000, 0.0367000],
        [0.0389000, -0.0685000, 1.0296000],
    ],
    "VonKries": [
        [0.4002400, 0.7076000, -0.0808100],
        [-0.2263000, 1.1653200, 0.0457000],
        [0.0000000, 0.0000000, 0.9182200],
    ],
}
scaling_matrixes_inv = {
    "XYZ": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    "Bradford": [
        [0.9869929, -0.1470543, 0.1599627],
        [0.4323053, 0.5183603, 0.0492912],
        [-0.0085287, 0.0400428, 0.9684867],
    ],
    "VonKries": [
        [1.8599364, -1.1293816, 0.2198974],
        [0.3611914, 0.6388125, -0.0000064],
        [0.0000000, 0.0000000, 1.0890636],
    ],
}


illuminant_colorimetry = pd.read_csv("data/illuminant.csv", index_col=None)


def illuminant_chromatic_adaptation_matrix(
    src: illuminant_space_names, dst: illuminant_space_names, scaling_method: scaling_methods = "Bradford"
) -> np.array:
    MA = np.array(scaling_matrixes[scaling_method], dtype=np.float64)
    MA_inv = np.array(scaling_matrixes_inv[scaling_method], dtype=np.float64)
    s_colorimetry = illuminant_colorimetry[illuminant_colorimetry["Illuminant"] == src]
    d_colorimetry = illuminant_colorimetry[illuminant_colorimetry["Illuminant"] == dst]

    WS = np.array(s_colorimetry.values[0, 1:], dtype=np.float64)
    WD = np.array(d_colorimetry.values[0, 1:], dtype=np.float64)

    rho_s, gamma_s, beta_s = MA @ WS
    rho_d, gamma_d, beta_d = MA @ WD

    CM = np.array([[rho_d / rho_s, 0, 0], [0, gamma_d / gamma_s, 0], [0, 0, beta_d / beta_s]])
    return MA @ CM @ MA_inv
