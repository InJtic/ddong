from __future__ import annotations
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np
from typing import Generator
from enum import Enum
from src.data.noise import NoiseGenerator


class Transition(ABC):
    @abstractmethod
    def iter(
        self, initial: NDArray[np.uint8]
    ) -> Generator[NDArray[np.uint8], None, None]: ...


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    UP_LEFT = (-1, -1)
    UP_RIGHT = (1, -1)
    DOWN_LEFT = (-1, 1)
    DOWN_RIGHT = (1, 1)

    def __add__(self, other: Direction):
        return Direction(
            (
                self.value[0] + other.value[0],
                self.value[1] + other.value[1],
            )
        )


class LinearTransition(Transition):
    def __init__(self, direction: Direction, total_frames: int, *, mpf: int = 1):
        # mpf = move per frame
        self.direction = direction
        self.total_frames = total_frames
        self.mpf = mpf

    def iter(self, initial: NDArray[np.uint8]):
        now = initial.copy()
        yield now

        for _ in range(self.total_frames):
            dx, dy = self.direction.value
            now = np.roll(a=now, shift=(self.mpf * dy, self.mpf * dx), axis=(0, 1))
            yield now


class NoTransition(Transition):
    def __init__(self, total_frames: int):
        self.total_frames = total_frames

    def iter(self, initial: NDArray[np.uint8]):
        now = initial.copy()
        for _ in range(self.total_frames):
            yield now


class NoiseOnly(Transition):
    def __init__(
        self,
        noise_generator: NoiseGenerator,
        total_frames: int,
        width: int,
        height: int,
    ):
        self.noise_generator = noise_generator
        self.total_frames = total_frames
        self.width = width
        self.height = height

    def iter(self, initial: NDArray[np.uint8]):
        yield initial.copy()

        for _ in range(self.total_frames):
            yield self.noise_generator(width=self.width, height=self.height)
