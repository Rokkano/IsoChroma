from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING

import numpy as np

from .abstract import CylindricalColorSpace

if TYPE_CHECKING:
    from .rgb import RGB


class HSV(CylindricalColorSpace):
    @staticmethod
    def from_rgb(rgb: RGB) -> HSV:
        R, G, B = rgb.values
        M = max(R, G, B)
        m = min(R, G, B)
        C = M - m

        if C == 0:
            H = 0
        elif M == R:
            H = ((G - B) / C) + (6 if G < B else 0)
        elif M == G:
            H = ((B - R) / C) + 2
        elif M == B:
            H = ((R - G) / C) + 4
        H /= 6  # Hue defined in [0;1]

        V = M

        S = 0 if V == 0 else C / V

        return HSV(np.array([H, S, V]))

    def to_rgb(self) -> RGB:
        H, S, V = self.values

        i = round(H * 6)
        f = H * 6 - i
        p = V * (1 - S)
        q = V * (1 - f * S)
        t = V * (1 - (1 - f) * S)

        if i % 6 == 0:
            R, G, B = V, t, p
        elif i % 6 == 1:
            R, G, B = q, V, p
        elif i % 6 == 2:
            R, G, B = p, V, t
        elif i % 6 == 3:
            R, G, B = p, q, V
        elif i % 6 == 4:
            R, G, B = t, p, V
        elif i % 6 == 5:
            R, G, B = V, p, q

        from .rgb import RGB

        return RGB(np.array([R, G, B]))


class HSVstd(CylindricalColorSpace):
    @staticmethod
    def from_rgb(rgb: RGB) -> HSVstd:
        R, G, B = rgb.values
        return HSVstd(np.array(colorsys.rgb_to_hsv(R, G, B)))

    def to_rgb(self) -> RGB:
        H, S, B = self.values
        from src.spaces.rgb import RGB

        return RGB(np.array(colorsys.hsv_to_rgb(H, S, B)))
