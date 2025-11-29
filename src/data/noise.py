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

    def __repr__(self):
        return f"BernolliNoise({self.p})"


class GaussianNoise(NoiseGenerator):
    def __init__(self, mean: float = 127, std: float = 20):
        """가우시안 노이즈

        평균이 `mean`, 표준편차가 `std`인 정규분포를 따르는 값으로 전체 픽셀을 칠합니다.

        Args:
            mean (float): 정규분포의 평균
            std (float): 정규분포의 표준편차
        """
        self.mean = mean
        self.std = std

    def __call__(self, width: int, height: int) -> NDArray[np.uint8]:
        noise = np.random.normal(self.mean, self.std, size=(height, width, 3))
        noise = np.clip(noise, 0, 255)
        return noise.astype(np.uint8)

    def __repr__(self):
        return f"GaussianNoise({self.mean}, {self.std})"
