from dataclasses import dataclass
from PIL import ImageFont
from src.transition import Transition
from numpy.typing import NDArray
import numpy as np
from data.noise import NoiseGenerator
from typing import Iterable, Callable, TypeAlias
from itertools import product, repeat, islice


Position: TypeAlias = tuple[float, float]


@dataclass
class TextInfo:
    text: str
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    position: Position
    initial: NDArray[np.uint8]
    transition: Transition


@dataclass
class BackgroundInfo:
    initial: NDArray[np.uint8]
    transition: Transition


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: int  # 단위: frame/seconds
    length: float  # 단위: seconds


@dataclass
class DataInfo:
    text_info: TextInfo
    background_info: BackgroundInfo
    video_info: VideoInfo


PositionBuilder = Callable[
    [
        str,  # text
        ImageFont.ImageFont | ImageFont.FreeTypeFont,  # font
    ],
    Position,
]


@dataclass
class DataGenerationConfig:
    # Text / Background Info
    text: str | Iterable[str]
    font: (
        ImageFont.ImageFont
        | ImageFont.FreeTypeFont
        | Iterable[ImageFont.ImageFont | ImageFont.FreeTypeFont]
    )
    text_position: Position | Callable[..., Position]
    n_position_sample: int
    noise_generator: NoiseGenerator
    text_transition: Transition | Iterable[Transition]
    background_transition: Transition | Iterable[Transition]

    # Video Info
    width: int
    height: int
    fps: int
    length: float

    def build(self) -> list[DataInfo]:
        """만들어질 데이터의 정보를 담은 리스트를 반환합니다."""

        # =====[TextInfo]=====
        text_info: list[TextInfo] = []

        # 텍스트 설정
        if isinstance(self.text, str):
            texts = [self.text]
        else:
            texts = self.text

        # 폰트 설정
        if isinstance(self.font, (ImageFont.ImageFont, ImageFont.FreeTypeFont)):
            fonts = [self.font]
        else:
            fonts = self.font

        # 포지션 결정

        if callable(self.text_position):
            positions = islice(
                iter(self.text_position, object()), self.n_position_sample
            )
        else:
            positions = repeat(self.text_position, self.n_position_sample)

        # 전환기 설정
        if isinstance(self.text_transition, Transition):
            text_transition = [self.text_transition]
        else:
            text_transition = self.text_transition

        for text, transition, position, font in product(
            texts,
            text_transition,
            positions,
            fonts,
        ):
            text_info.append(
                TextInfo(
                    text=text,
                    font=font,
                    position=position,
                    initial=self.noise_generator(width=self.width, height=self.height),
                    transition=transition,
                )
            )

        # =====[BackgroundInfo]=====
        background_info: list[BackgroundInfo] = []

        # 전환기 설정
        if isinstance(self.background_transition, Transition):
            background_transition = [self.background_transition]
        else:
            background_transition = self.background_transition

        for transition in background_transition:
            background_info.append(
                BackgroundInfo(
                    initial=self.noise_generator(width=self.width, height=self.height),
                    transition=transition,
                )
            )

        # =====[VideoInfo]=====
        video_info: list[VideoInfo] = [
            VideoInfo(
                width=self.width,
                height=self.height,
                fps=self.fps,
                length=self.length,
            )
        ]

        return list(
            map(
                lambda info: DataInfo(*info),
                product(text_info, background_info, video_info),
            )
        )
