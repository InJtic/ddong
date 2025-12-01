from src.transition import Transition, LinearTransition, Direction, NoTransition
from src.config import DataGenerationConfig
from src.utils import (
    get_centered_position,
    get_position_builder,
    get_sized_fonts,
    save_metadata,
    Metadata,
)
from src.data.noise import BernoulliNoise
from src.data.generator import DataGenerator
from itertools import product
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
from typing import NamedTuple
from src.config import Position
from typing import Callable
from src.utils import cleanup
from dataclasses import dataclass


class ProcessArg(NamedTuple):
    index: int
    speed: int
    direction: Direction
    label: str
    font_size: float
    fps: int
    position: Callable[..., Position] | Callable[..., Callable[..., Position]]
    noise_level: float


@dataclass
class MetadataWithPosition(Metadata):
    position: str


def process(arg: ProcessArg):
    (index, speed, direction, label, font_size, fps, position, noise_level) = arg
    n_position_sample = 9
    binary = ("0", "1")
    quad = ("A", "B", "C", "D")

    directory = f"data/center_vs_random/{index}"

    np.random.seed(index)

    transition = LinearTransition(direction=direction, total_frames=fps * 2, mpf=speed)
    font_path = "resources/malgun.ttf"
    position_eval = position(
        width=224,
        height=224,
        text=label,
        font=get_sized_fonts(
            width=224,
            font_path=font_path,
            text=label,
            percent=font_size,
        ),
    )

    execute(
        text_transition=transition,
        text=label,
        font_size_percent=font_size,
        n_position_sample=n_position_sample,
        fps=fps,
        directory=directory,
        position=position_eval,
        noise_level=noise_level,
    )

    return MetadataWithPosition(
        move_per_frame=speed,
        move_direction=direction.name,
        label=label,
        options=binary if label in binary else quad,
        fps=fps,
        font_size=font_size,
        length=2,
        width=224,
        height=224,
        noise=f"BernoulliNoise({noise_level})",
        seed=index,
        savedat=directory,
        position="random" if callable(position_eval) else "centered",
    )


def execute(
    text_transition: Transition,
    text: str,
    font_size_percent: float,
    n_position_sample: int,
    fps: int,
    directory: str,
    position: Position | Callable[..., Position],
    noise_level: float,
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
    length = 2
    total_frames = fps * length
    noise_generator = BernoulliNoise(noise_level)

    info = DataGenerationConfig(
        text=text,
        font=font,
        text_position=position,
        noise_generator=noise_generator,
        text_transition=text_transition,
        n_position_sample=n_position_sample,
        background_transition=NoTransition(total_frames=total_frames),
        width=width,
        height=height,
        fps=fps,
        length=length,
        text_fill=False,
    ).build()

    data_generator = DataGenerator(info)

    for video in data_generator:
        video.save(directory)


def main():
    speeds = (1,)  # 3, 7)
    directions = (Direction.DOWN,)  # Direction.UP_RIGHT)
    labels = tuple("01ABCD")
    tasks = []
    font_sizes = (0.2,)  # 0.4, 0.6)
    fpss = (10,)  # 20, 30)

    for i, (
        position,
        speed,
        direction,
        label,
        font_size,
        fps,
        noise_level,
    ) in enumerate(
        product(
            (get_centered_position, get_position_builder),
            speeds,
            directions,
            labels,
            font_sizes,
            fpss,
            (1, 0.999, 0.99, 0.9, 0.8),
        )
    ):
        tasks.append(
            (i, speed, direction, label, font_size, fps, position, noise_level)
        )

    metadata_path = "data/center_vs_random/metadata.csv"

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
    cleanup("data/center_vs_random/metadata.csv")
