"""
이동 속도 : (1, 3, 7)
전환 방법 : (LinearTransform(direction.DOWN), LinearTransform(direction.UP_RIGHT))
텍스트 : ["0", "1", "A", "B", "C", "D"]
레이블 당 위치 수 : 25
폰트 사이즈 : (0.2, 0.4, 0.6)
FPS : (10, 20, 30)
"""

from src.transition import Transition, LinearTransition, Direction, NoTransition
from src.config import DataGenerationConfig
from src.utils import get_position_builder, get_sized_fonts, save_metadata, Metadata
from src.data.noise import BernoulliNoise
from src.data.generator import DataGenerator
from itertools import product
import numpy as np


def execute(
    text_transition: Transition,
    text: str,
    font_size_percent: float,
    n_position_sample: int,
    fps: int,
    directory: str,
):
    width = 224
    height = 224
    font_path = "resources/malgun.ttf"

    font = get_sized_fonts(
        width=width,
        font_path=font_path,
        text=text,
        percent=font_size_percent,
    )
    position_builder = get_position_builder(
        text=text,
        font=font,
        width=width,
        height=height,
    )
    length = 2
    total_frames = fps * length
    noise_generator = BernoulliNoise(0.8)

    info = DataGenerationConfig(
        text=text,
        font=font,
        text_position=position_builder,
        noise_generator=noise_generator,
        text_transition=text_transition,
        n_position_sample=n_position_sample,
        background_transition=NoTransition(total_frames=total_frames),
        width=width,
        height=height,
        fps=fps,
        length=length,
    ).build()

    data_generator = DataGenerator(info)

    for video in data_generator:
        video.save(directory)


def main():
    speeds = (1, 3, 7)
    directions = (Direction.DOWN, Direction.UP_RIGHT)
    labels = tuple("01ABCD")
    n_position_sample = 25
    font_sizes = (0.2, 0.4, 0.6)
    fpss = (10, 20, 30)
    binary = ("0", "1")
    quad = ("A", "B", "C", "D")

    for i, (
        speed,
        direction,
        label,
        font_size,
        fps,
    ) in enumerate(
        product(
            speeds,
            directions,
            labels,
            font_sizes,
            fpss,
        )
    ):
        np.random.seed(i)
        transition = LinearTransition(
            direction=direction, total_frames=fps * 2, mpf=speed
        )

        execute(
            text_transition=transition,
            text=label,
            font_size_percent=font_size,
            n_position_sample=n_position_sample,
            fps=fps,
            directory=f"data/{i}",
        )

        save_metadata(
            path="data/metadata.csv",
            metadata=Metadata(
                move_per_frame=speed,
                move_direction=direction.name,
                label=label,
                options=binary if label in binary else quad,
                fps=fps,
                font_size=font_size,
                length=2,
                width=224,
                height=224,
                noise="BernoulliNoise(0.8)",
                seed=i,
                savedat=f"data/{i}",
            ),
        )
