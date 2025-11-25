from numpy.typing import NDArray
import numpy as np
from abc import ABC, abstractmethod


class NoiseGenerator(ABC):
    @abstractmethod
    def __call__(self, width: int, height: int) -> NDArray[np.uint8]:
        """`(width, height)` 사이즈의 노이즈를 생성합니다."""


class BernoulliNoise(NoiseGenerator):
    def __init__(self, p: float):
        """베르누이 노이즈

        0 또는 255로 전체 픽셀을 칠합니다.

        Args:
            p (float): 픽셀을 검은색(0)으로 칠할 확률
        """
        self.p = p

    def __call__(self, width: int, height: int) -> NDArray[np.uint8]:
        return np.random.choice(
            [0, 255],
            size=(height, width),
            p=[self.p, 1 - self.p],
        ).astype(np.uint8)
