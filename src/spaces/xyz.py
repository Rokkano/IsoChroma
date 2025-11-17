from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..chromatic_adaptation import illuminant_chromatic_adaptation_matrix
from .abstract import CartesianColorSpace
from .rgb import rgb_colorimetry, rgb_colorimetry_space_names, whites_colorimetry

if TYPE_CHECKING:
    from .lms import LMS
    from .oklab import OKLAB
    from .rgb import RGB


class XYZ(CartesianColorSpace):
    @staticmethod
    def from_rgb(
        rgb: RGB, rgb_space_name: rgb_colorimetry_space_names = "sRGB", bradford_adapted_d50: bool = True
    ) -> XYZ:
        M = rgb_to_xyz_matrix(rgb_space_name)
        if bradford_adapted_d50:
            w_ref = rgb_colorimetry[rgb_colorimetry["Name"] == rgb_space_name]["Reference White"].item()
            if w_ref != "D50":
                BFM = illuminant_chromatic_adaptation_matrix(w_ref, "D50", "Bradford")
                M = BFM @ M
        return XYZ(M @ rgb.values)

    def to_rgb(self, rgb_space_name: rgb_colorimetry_space_names = "sRGB", bradford_adapted_d50: bool = True) -> RGB:
        if len(self.values) != 3:
            raise ValueError("Argument should be a 3 floating point value numpy array.")

        M = rgb_to_xyz_matrix(rgb_space_name)
        if bradford_adapted_d50:
            w_ref = rgb_colorimetry[rgb_colorimetry["Name"] == rgb_space_name]["Reference White"].item()
            if w_ref != "D50":
                BFM = illuminant_chromatic_adaptation_matrix(w_ref, "D50", "Bradford")
                M = BFM @ M
        from .rgb import RGB

        return RGB(np.linalg.inv(M) @ self.values)

    @staticmethod
    def from_lms(lms: LMS) -> XYZ:
        return lms.to_xyz()

    def to_lms(self) -> LMS:
        from .lms import LMS

        return LMS.from_xyz(self)

    @staticmethod
    def from_oklab(oklab: OKLAB) -> XYZ:
        return oklab.to_xyz()

    def to_oklab(self) -> OKLAB:
        from .oklab import OKLAB

        return OKLAB.from_xyz(self)


def rgb_to_xyz_matrix(rgb_space_name: rgb_colorimetry_space_names = "sRGB") -> np.ndarray[float]:
    rgb_space = rgb_colorimetry[rgb_colorimetry["Name"] == rgb_space_name]
    xr = pd.to_numeric(rgb_space["Red Primary x"].item())
    yr = pd.to_numeric(rgb_space["Red Primary y"].item())
    xg = pd.to_numeric(rgb_space["Green Primary x"].item())
    yg = pd.to_numeric(rgb_space["Green Primary y"].item())
    xb = pd.to_numeric(rgb_space["Blue Primary x"].item())
    yb = pd.to_numeric(rgb_space["Blue Primary y"].item())

    Xr = xr / yr
    Yr = 1
    Zr = (1 - xr - yr) / yr
    Xg = xg / yg
    Yg = 1
    Zg = (1 - xg - yg) / yg
    Xb = xb / yb
    Yb = 1
    Zb = (1 - xb - yb) / yb

    W = np.array(
        whites_colorimetry[whites_colorimetry["Illuminant"] == rgb_space["Reference White"].item()].values[0, 1:],
        dtype=np.float64,
    )

    XYZ = np.array(
        [
            [Xr, Xg, Xb],
            [Yr, Yg, Yb],
            [Zr, Zg, Zb],
        ]
    )

    S = np.linalg.inv(XYZ) @ W
    M = S * XYZ

    return M
