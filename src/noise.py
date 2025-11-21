from abc import ABC, abstractmethod
from typing import TypeAlias
import numpy as np

_NoiseArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.uint8]]


class Noise(ABC):
    @abstractmethod
    def fill(self, width: int, height: int) -> _NoiseArray:
        """노이즈로 채운 배열을 반환합니다.

        Args:
            width (int): 가로
            height (int): 세로

        Returns:
            _NoiseArray: 노이즈로 채워진 배열. 각각은 `[0, 255]`의 수임.
        """


class BernoulliNoise(Noise):
    def __init__(self, prob: float):
        """베르누이 노이즈

        각 위치에 대해 `0, 255`중 하나만 선택합니다.

        Args:
            prob (float): `0`(검은색)을 선택할 확률
        """
        assert 0 <= prob <= 1
        self.prob = prob

    def fill(self, width: int, height: int) -> _NoiseArray:
        noise_arr = np.random.choice(
            [0, 255], size=(height, width), p=[self.prob, 1 - self.prob]
        ).astype(np.uint8)

        return noise_arr
