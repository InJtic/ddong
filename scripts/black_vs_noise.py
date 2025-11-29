from src.transition import Transition, LinearTransition, Direction, NoTransition
from src.config import DataGenerationConfig
from src.utils import get_centered_position, get_sized_fonts, save_metadata, Metadata
from src.data.noise import BernoulliNoise
from src.data.generator import DataGenerator
from itertools import product
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
from typing import NamedTuple
from dataclasses import dataclass


@dataclass
class MetadataWithFill(Metadata):
    fill: bool


class ProcessArg(NamedTuple):
    index: int
    speed: int
    direction: Direction
    label: str
    font_size: float
    fps: int
    text_fill: bool


def process(arg: ProcessArg):
    (
        index,
        speed,
        direction,
        label,
        font_size,
        fps,
        text_fill,
    ) = arg
    n_position_sample = 9
    binary = ("0", "1")
    quad = ("A", "B", "C", "D")

    directory = f"data/black_vs_noise/{index}"

    np.random.seed(index)

    transition = LinearTransition(direction=direction, total_frames=fps * 2, mpf=speed)

    execute(
        background_transition=transition,
        text=label,
        font_size_percent=font_size,
        n_position_sample=n_position_sample,
        fps=fps,
        directory=directory,
        text_fill=text_fill,
    )

    return MetadataWithFill(
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
        seed=index,
        savedat=directory,
        fill=text_fill,
    )


def execute(
    background_transition: Transition,
    text: str,
    font_size_percent: float,
    n_position_sample: int,
    fps: int,
    directory: str,
    text_fill: bool,
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
    position = get_centered_position(
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
        text_position=position,
        noise_generator=noise_generator,
        text_transition=NoTransition(total_frames=total_frames),
        n_position_sample=n_position_sample,
        background_transition=background_transition,
        width=width,
        height=height,
        fps=fps,
        length=length,
        text_fill=text_fill,
    ).build()

    data_generator = DataGenerator(info)

    for video in data_generator:
        video.save(directory)


def main():
    speeds = (1, 3, 7)
    directions = (Direction.DOWN, Direction.UP_RIGHT)
    labels = tuple("01ABCD")
    tasks = []
    font_sizes = (0.2, 0.4, 0.6)
    fpss = (10, 20, 30)

    for i, (
        text_fill,
        speed,
        direction,
        label,
        font_size,
        fps,
    ) in enumerate(
        product(
            (True, False),
            speeds,
            directions,
            labels,
            font_sizes,
            fpss,
        )
    ):
        tasks.append((i, speed, direction, label, font_size, fps, text_fill))

    metadata_path = "data/black_vs_noise/metadata.csv"

    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    print(f"Total Tasks: {len(tasks)}")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(tasks)):
            metadata = future.result()

            save_metadata(path=metadata_path, metadata=metadata)


if __name__ == "__main__":
    main()
