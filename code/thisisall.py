from abc import ABC, abstractmethod
from typing import TypeAlias
import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image

_Frame: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.uint8]]
_Mask: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.bool]]


class Noise(ABC):
    @abstractmethod
    def fill(
        self,
        shape: tuple[int, ...],
        *,
        mask: _Mask | None = None,
    ) -> _Frame: ...


class BernoulliNoise(Noise):
    def __init__(self, prob: float):
        assert 0 <= prob <= 1
        self.prob = prob

    def fill(
        self,
        shape: tuple[int, ...],
        *,
        mask: _Mask | None = None,
    ):
        frame: _Frame = np.random.choice(
            [0, 255], size=shape, p=[self.prob, 1 - self.prob]
        ).astype(np.uint8)

        if mask is not None:
            frame *= ~mask

        return frame


def get_text_mask(
    shape: tuple[int, int],
    text: str,
    font_path: str | None = None,
    font_size: int = 150,
) -> _Mask:
    mask = Image.new("L", shape, 0)
    draw = ImageDraw.Draw(mask)
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default(size=font_size)
    else:
        font = ImageFont.load_default(size=font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    x = (shape[0] - width) // 2
    y = (shape[1] - height) // 2

    draw.text((x, y), text, font=font, fill=255)

    return (np.array(mask) / 255).astype(np.bool)


class VideoGenerator:
    def __init__(
        self,
        width: int,
        height: int,
        noise: Noise,
        text: str,
        font_path: str | None = None,
        font_size: int = 150,
    ):
        self.noise = noise
        self.frames = [noise.fill(shape=(height, width))]
        self.base = noise.fill(shape=(height, width))
        self.mask = get_text_mask(
            (width, height), text=text, font_path=font_path, font_size=font_size
        )
        self.frames[0][self.mask] = self.base[self.mask]

    def step(self):
        prev = self.frames[-1]
        self.base = np.roll(self.base, 1, axis=1)

        new = prev.copy()
        new[self.mask] = self.base[self.mask]
        self.frames.append(new)

    def save(self, path: str, fps: int = 30):
        fourcc = cv2.VideoWriter.fourcc(*"XVID")
        height, width = self.frames[0].shape
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

        for frame in self.frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(bgr)

        writer.release()


if __name__ == "__main__":
    vg = VideoGenerator(
        width=640,
        height=480,
        noise=BernoulliNoise(prob=0.8),
        text="HELLO",
    )

    for _ in range(200):
        vg.step()

    vg.save(path="./test.avi", fps=30)
