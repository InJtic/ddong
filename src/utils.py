from src.config import DataGenerationConfig, Position
from PIL import ImageFont, Image, ImageDraw
import numpy as np
from numpy.typing import NDArray
from typing import Callable
from src.data.noise import BernoulliNoise
from src.transition import LinearTransition, NoTransition, Direction, NoiseOnly
from dataclasses import dataclass, asdict
import pandas as pd
import os
from pathlib import Path


def get_sized_fonts(
    width: int,
    font_path: str | None,
    text: str | list[str],
    percent: float,
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    if font_path is None:
        font = ImageFont.load_default(100)
    else:
        font = ImageFont.truetype(font_path, 100)

    if isinstance(text, str):
        text = [text]

    text_width = font.getlength(max(text, key=lambda t: font.getlength(t)))

    if font_path is None:
        new_font = ImageFont.load_default(int(100 * width * percent / text_width))
    else:
        new_font = ImageFont.truetype(
            font_path,
            int(100 * width * percent / text_width),
        )

    return new_font


def load_default_data_settings() -> DataGenerationConfig:
    width = 224
    height = 224
    text = "0"
    font_path = "resources/malgun.ttf"

    font = get_sized_fonts(
        width=width,
        font_path=font_path,
        text=text,
        percent=0.3,
    )
    position_builder = get_position_builder(
        text=text,
        font=font,
        width=width,
        height=height,
    )
    fps = 10
    length = 2
    total_frames = fps * length
    noise_generator = BernoulliNoise(0.8)

    return DataGenerationConfig(
        text="0",
        font=font,
        text_position=position_builder,
        noise_generator=noise_generator,
        text_transition=(
            LinearTransition(
                direction=Direction.DOWN + Direction.LEFT,
                total_frames=total_frames,
                mpf=1,
            ),
        ),
        n_position_sample=25,
        background_transition=(
            NoTransition(total_frames=total_frames),
            NoiseOnly(
                noise_generator=noise_generator,
                total_frames=total_frames,
                width=width,
                height=height,
            ),
        ),
        width=width,
        height=height,
        fps=fps,
        length=length,
    )


def get_position_builder(
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    width: int,
    height: int,
) -> Callable[..., Position]:
    text_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(text_image)
    bbox = draw.textbbox((0, 0), text, font=font)

    # 왼쪽 위를 기준으로 계산
    x_min = -bbox[0]
    x_max = width - bbox[2]
    y_min = -bbox[1]
    y_max = height - bbox[3]

    return lambda: (
        np.random.rand() * (x_max - x_min) + x_min,
        np.random.rand() * (y_max - y_min) + y_min,
    )


def get_text_mask(
    position: Position,
    width: int,
    height: int,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> NDArray[np.bool]:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.text(xy=position, text=text, font=font, fill=1)
    return np.array(mask).astype(np.bool)


@dataclass
class Metadata:
    move_per_frame: int
    move_direction: str
    label: str
    options: tuple[str, ...]
    fps: int
    font_size: float
    length: int
    width: int
    height: int
    noise: str
    seed: int
    savedat: str


def save_metadata(
    path: str,
    metadata: Metadata,
):
    data = asdict(metadata)

    df = pd.DataFrame([data])

    if not os.path.isfile(path):
        pathobj = Path(path)
        pathobj.mkdir(parents=True, exist_ok=True)
        pathobj.touch()

    df.to_csv(path, mode="a", index=True)
