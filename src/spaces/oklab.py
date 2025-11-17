from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.spaces.abstract import CartesianColorSpace

if TYPE_CHECKING:
    from src.spaces.rgb import RGB

    from .xyz import XYZ

Oklab_LMS_matrix = [
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070],
]

Oklab_matrix = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.80806757660],
]


class OKLAB(CartesianColorSpace):
    @staticmethod
    def from_xyz(xyz: XYZ):
        lms = Oklab_LMS_matrix @ xyz.values
        lmsp = lms ** (1 / 3)
        lab = Oklab_matrix @ lmsp
        return OKLAB(lab)

    def to_xyz(self):
        lmsp = np.linalg.inv(Oklab_matrix) @ self.values
        lms = lmsp**3
        xyz = np.linalg.inv(Oklab_LMS_matrix) @ lms
        from src.spaces.xyz import XYZ

        return XYZ(xyz)

    @staticmethod
    def from_rgb(rgb: RGB) -> OKLAB:
        from .xyz import XYZ

        return OKLAB.from_xyz(XYZ.from_rgb(rgb))

    def to_rgb(self) -> RGB:
        from .xyz import XYZ

        return XYZ.to_rgb(OKLAB.to_xyz(self))
