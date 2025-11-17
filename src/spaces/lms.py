from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .abstract import CartesianColorSpace
from .rgb import rgb_colorimetry_space_names

if TYPE_CHECKING:
    from .rgb import RGB
    from .xyz import XYZ

EEI_matrix = [
    [0.38971, 0.68898, -0.07868],
    [-0.22981, 1.18640, 0.04641],
    [0, 0, 1],
]


class LMS(CartesianColorSpace):
    @staticmethod
    def from_xyz(xyz: XYZ) -> LMS:
        return LMS(np.array(EEI_matrix) @ xyz.values)

    def to_xyz(self) -> XYZ:
        from .xyz import XYZ

        return XYZ(np.linalg.inv(np.array(EEI_matrix)) @ self.values)

    @staticmethod
    def from_rgb(
        rgb: RGB, rgb_space_name: rgb_colorimetry_space_names = "sRGB", bradford_adapted_d50: bool = True
    ) -> LMS:
        return LMS.from_xyz(XYZ.from_rgb(rgb, rgb_space_name, bradford_adapted_d50))

    def to_rgb(self, rgb_space_name: rgb_colorimetry_space_names = "sRGB", bradford_adapted_d50: bool = True) -> RGB:
        return self.to_xyz().to_rgb(rgb_space_name, bradford_adapted_d50)
