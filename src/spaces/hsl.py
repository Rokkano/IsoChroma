from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING

import numpy as np

from .abstract import CylindricalColorSpace

if TYPE_CHECKING:
    from .rgb import RGB


class HSL(CylindricalColorSpace):
    @staticmethod
    def from_rgb(rgb: RGB) -> HSL:
        R, G, B = rgb.values
        M = max(R, G, B)
        m = min(R, G, B)
        C = M - m

        if C == 0:
            H = 0
        elif M == R:
            H = ((G - B) / C) % 6
        elif M == G:
            H = ((B - R) / C) + 2
        elif M == B:
            H = ((R - G) / C) + 4
        else:
            raise ValueError()
        H /= 6  # Hue defined in [0;1]

        L = 1 / 2 * C

        S = 0 if L == 1 or L == 0 else C / (1 - abs(2 * L - 1))

        return HSL(np.array([H, S, L]))

    def to_rgb(self) -> RGB:
        H, S, L = self.values
        if S == 0:
            R = G = B = L  # achromatic
        else:
            q = L * (1 + S) if L < 0.5 else L + S - L * S
            p = 2 * L - q
            R = self._hue_to_rgb(p, q, H + 1 / 3)
            G = self._hue_to_rgb(p, q, H)
            B = self._hue_to_rgb(p, q, H - 1 / 3)
        from .rgb import RGB

        return RGB(np.array([R, G, B]))

    def _hue_to_rgb(self, p: float, q: float, t: float) -> float:
        if t < 0:
            t += 1
        elif t > 1:
            t -= 1
        elif t < 1 / 6:
            return p + (q - p) * 6 * t
        elif t < 1 / 2:
            return q
        elif t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p


class HSLstd(CylindricalColorSpace):
    @staticmethod
    def from_rgb(rgb: RGB) -> HSLstd:
        R, G, B = rgb.values
        H, L, S = colorsys.rgb_to_hls(R, G, B)
        return HSLstd(np.array([H, S, L]))

    def to_rgb(self) -> RGB:
        H, S, L = self.values
        from src.spaces.rgb import RGB

        return RGB(np.array(colorsys.hls_to_rgb(H, L, S)))
