from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from .abstract import CartesianColorSpace, ColorSpace

if TYPE_CHECKING:
    from .hsl import HSL, HSLstd
    from .hsv import HSV, HSVstd
    from .lms import LMS
    from .oklab import OKLAB
    from .xyz import XYZ

rgb_colorimetry_space_names = Literal[
    "Lab Gamut",
    "Adobe RGB (1998)",
    "Apple RGB",
    "Best RGB",
    "Beta RGB",
    "Bruce RGB",
    "CIE RGB",
    "ColorMatch RGB",
    "Don RGB 4",
    "ECI RGB v2",
    "Ekta Space PS5",
    "NTSC RGB",
    "PAL/SECAM RGB",
    "ProPhoto RGB",
    "SMPTE-C RGB",
    "sRGB",
    "Wide Gamut RGB",
]
illuminant_space_names = Literal["A", "B", "C", "D50", "D55", "D65", "D75", "E", "F2", "F7", "F11"]

rgb_colorimetry = pd.read_csv("data/rgb.csv", index_col=None)
whites_colorimetry = pd.read_csv("data/illuminant.csv", index_col=None)


class RGB(CartesianColorSpace):
    @staticmethod
    def from_xyz(
        xyz: XYZ, rgb_space_name: rgb_colorimetry_space_names = "sRGB", bradford_adapted_d50: bool = True
    ) -> RGB:
        return xyz.to_rgb(rgb_space_name, bradford_adapted_d50)

    def to_xyz(self, rgb_space_name: rgb_colorimetry_space_names = "sRGB", bradford_adapted_d50: bool = True) -> XYZ:
        from .xyz import XYZ

        return XYZ.from_rgb(self, rgb_space_name, bradford_adapted_d50)

    @staticmethod
    def from_lms(
        lms: LMS, rgb_space_name: rgb_colorimetry_space_names = "sRGB", bradford_adapted_d50: bool = True
    ) -> RGB:
        return lms.to_xyz().to_rgb(rgb_space_name, bradford_adapted_d50)

    def to_lms(self, rgb_space_name: rgb_colorimetry_space_names = "sRGB", bradford_adapted_d50: bool = True) -> LMS:
        from .lms import LMS
        from .xyz import XYZ

        return LMS.from_xyz(XYZ.from_rgb(self, rgb_space_name, bradford_adapted_d50))

    @staticmethod
    def from_rgb255(rgb255: RGB255) -> RGB:
        return RGB255.to_rgb(rgb255)

    def to_rgb255(self) -> RGB255:
        return RGB255.from_rgb(self)

    @staticmethod
    def from_hex(hex: HEX) -> RGB:
        return hex.to_rgb()

    def to_hex(self):
        return HEX.from_rgb(self)

    @staticmethod
    def from_hsl(hsl: HSL) -> RGB:
        return hsl.to_rgb()

    def to_hsl(self) -> HSL:
        from .hsl import HSL

        return HSL.from_rgb(self)

    @staticmethod
    def from_hslstd(hslstd: HSLstd) -> RGB:
        return hslstd.to_rgb()

    def to_hslstd(self) -> HSLstd:
        from .hsl import HSLstd

        return HSLstd.from_rgb(self)

    @staticmethod
    def from_hsv(hsv: HSV) -> RGB:
        return hsv.to_rgb()

    def to_hsv(self) -> HSV:
        from .hsv import HSV

        return HSV.from_rgb(self)

    @staticmethod
    def from_hsvstd(hsvstd: HSVstd) -> RGB:
        return hsvstd.to_rgb()

    def to_hsvstd(self) -> HSVstd:
        from .hsv import HSVstd

        return HSVstd.from_rgb(self)

    @staticmethod
    def from_oklab(oklab: OKLAB) -> RGB:
        return OKLAB.to_rgb(oklab)

    def to_oklab(self) -> OKLAB:
        from .oklab import OKLAB

        return OKLAB.from_rgb(self)


class RGB255(CartesianColorSpace):
    @staticmethod
    def from_rgb(rgb: RGB) -> RGB255:
        return RGB255((rgb.values.clip(0, 1) * 255).round().astype(int))

    def to_rgb(self):
        return RGB((self.values // 255).astype(float))

    @staticmethod
    def from_hex(hex: HEX) -> RGB255:
        return hex.to_rgb255()

    def to_hex(self):
        return HEX.from_rgb255(self)

    def __str__(self):
        return f"({self.values[0]}, {self.values[1]}, {self.values[2]})"


class HEX(ColorSpace):
    @staticmethod
    def from_rgb(rgb: RGB) -> HEX:
        return HEX.from_rgb255(RGB255.from_rgb(rgb))

    def to_rgb(self):
        return self.to_rgb255().to_rgb()

    @staticmethod
    def from_rgb255(rgb255: RGB255) -> HEX:
        r, g, b = rgb255.values
        return HEX(["{:02x}".format(r), "{:02x}".format(g), "{:02x}".format(b)])

    def to_rgb255(self):
        return RGB255(tuple(map(lambda x: int(x, 16), self.values)))

    def __str__(self):
        return f"#{self.values[0]}{self.values[1]}{self.values[2]}".upper()
