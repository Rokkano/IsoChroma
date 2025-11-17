from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
from typing_extensions import Self


class ColorSpace(ABC):
    values: np.ndarray[Any]

    def __init__(self, values: np.ndarray[Any]):
        super().__init__()
        self.values = values

    def __getitem__(self, index: int) -> Any:
        if index > len(self.values):
            raise ValueError("Index out of range.")
        return self.values[index]

    def __setitem__(self, index: int, element: Any) -> None:
        if index > len(self.values):
            raise ValueError("Index out of range.")
        self.values[index] = element

    def __len__(self):
        return len(self.values)

    def get_iterator(self):
        return iter(self.values)


class CartesianColorSpace(ColorSpace, ABC):
    def __add__(self, other: Self | float) -> Self:
        if type(self) is type(other):
            return type(self)(self.values + other.values)
        elif type(other) is float:
            return type(self)(self.values + other)
        raise ValueError("Cannot use operation on different ColorSpace.")

    def __sub__(self, other: Self | float) -> Self:
        if type(self) is type(other):
            return type(self)(self.values - other.values)
        elif type(other) is float:
            return type(self)(self.values - other)
        raise ValueError("Cannot use operation on different ColorSpace.")

    def __mul__(self, other: Self | float) -> Self:
        if type(self) is type(other):
            return type(self)(self.values * other.values)
        elif type(other) is float:
            return type(self)(self.values * other)
        raise ValueError("Cannot use operation on different ColorSpace.")

    def __truediv__(self, other: Self | float) -> Self:
        if type(self) is type(other):
            return type(self)(self.values / other.values)
        elif type(other) is float:
            return type(self)(self.values / other)
        raise ValueError("Cannot use operation on different ColorSpace.")

    def __pow__(self, other: float) -> Self:
        return type(self)(self.values**other)

    def __neg__(self) -> Self:
        return type(self)(-self.values)

    def __abs__(self) -> Self:
        return type(self)(abs(self.values))


class CylindricalColorSpace(ColorSpace, ABC):
    @staticmethod
    def _cylindrical_to_cartesian(r: float, theta: float, z: float) -> np.ndarray[float]:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.array([x, y, z])

    @staticmethod
    def _cartesian_to_cylindrical(x: float, y: float, z: float) -> np.ndarray[float]:
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return np.array([r, theta, z])

    def __add__(self, other: Self) -> Self:
        if type(self) is not type(other):
            raise ValueError("Cannot use operation on different ColorSpace.")
        a = self._cylindrical_to_cartesian(*self.values)
        b = self._cylindrical_to_cartesian(*other.values)
        c = self._cartesian_to_cylindrical(*(a + b))
        return type(self)(c)

    def __sub__(self, other: Self) -> Self:
        if type(self) is not type(other):
            raise ValueError("Cannot use operation on different ColorSpace.")
        a = self._cylindrical_to_cartesian(*self.values)
        b = self._cylindrical_to_cartesian(*other.values)
        c = self._cartesian_to_cylindrical(*(a - b))
        return type(self)(c)

    def __neg__(self) -> Self:
        return type(self)(-self.values)

    def __abs__(self) -> Self:
        return type(self)(abs(self.values))
