from abc import ABC, abstractmethod
from copy import deepcopy
from src.noise import _NoiseArray
from enum import Enum
import numpy as np


class Transform(ABC):
    def __init__(self, initial: _NoiseArray):
        self.content = initial

    @abstractmethod
    def __next__(self) -> _NoiseArray: ...
    def __iter__(self):
        new = deepcopy(self)
        yield new.content
        while True:
            yield new.__next__()


class Direction(Enum):
    # (shift, axis)
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (-1, 1)
    RIGHT = (1, 1)


class LinearTransform(Transform):
    def __init__(self, initial: _NoiseArray, direction: Direction):
        self.shift, self.axis = direction.value
        self.content = initial

    def __next__(self) -> _NoiseArray:
        self.content = np.roll(
            self.content,
            shift=self.shift,
            axis=self.axis,
        )

        return self.content


class NoTransform(Transform):
    def __init__(self, initial: _NoiseArray):
        self.content = initial

    def __next__(self) -> _NoiseArray:
        return self.content
