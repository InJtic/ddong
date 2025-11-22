from itertools import islice
import math
import os
from typing import Iterable
from src import (
    Direction,
    VideoGenerator,
    NoTransform,
    BernoulliNoise,
    LinearTransform,
)
from src.video import _Mask
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random


def _get_random_text_mask(
    width: int,
    height: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> _Mask:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # 1. 텍스트의 Bounding Box 구하기 (좌표는 0,0 기준)
    bbox = draw.textbbox((0, 0), text, font=font)

    # 2. 텍스트의 실제 너비/높이 및 오프셋 계산
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    offset_x = bbox[0]
    offset_y = bbox[1]

    # 3. 텍스트가 움직일 수 있는 최대 좌표 (Safe Zone) 계산
    # max 값이 0보다 작으면(글자가 이미지보다 크면) 0으로 고정
    max_x = max(0, math.floor(width - text_width))
    max_y = max(0, math.floor(height - text_height))

    # 4. 랜덤 좌표 생성 (이미지 내부에 들어갈 Top-Left 좌표)
    random_x = random.randint(0, max_x)
    random_y = random.randint(0, max_y)

    # 5. 그리기 (오프셋 보정 포함)
    # bbox[0], bbox[1]을 빼주어야 실제 픽셀이 random_x, random_y에서 시작됨
    final_x = random_x - offset_x
    final_y = random_y - offset_y

    draw.text((final_x, final_y), text, font=font, fill=1)

    return np.array(mask).astype(bool)


def main(
    n_samples: int,
    direction: Direction,
    labels: Iterable[str],
    speed: int = 1,
    mask_maker=_get_random_text_mask,
):
    noise = BernoulliNoise(0.8)

    vg = VideoGenerator(
        width=224,
        height=224,
        backgrounds=islice(
            NoTransform(initial=noise.fill(width=224, height=224)),
            0,
            30 * speed * 2,
            speed,
        ),
        texts=islice(
            LinearTransform(
                initial=noise.fill(width=224, height=224), direction=direction
            ),
            0,
            30 * speed * 2,
            speed,
        ),
        font_path="./resources/malgun.ttf",
        font_size=100,
        fps=30,
    )

    os.makedirs("data", exist_ok=True)
    os.makedirs(f"data/{speed}_{direction.name}", exist_ok=True)

    for c in labels:
        os.makedirs(f"data/{speed}_{direction.name}/{c}", exist_ok=True)
        for sample in range(n_samples):
            raw = vg.raw(text=c, mask_maker=mask_maker)
            for frame, image in enumerate(raw):
                image.save(
                    f"data/{speed}_{direction.name}/{c}/{sample + 1}_{frame + 1:03}.png"
                )
            vg.init_transforms(
                backgrounds=islice(
                    NoTransform(initial=noise.fill(width=224, height=224)),
                    0,
                    30 * speed * 2,
                    speed,
                ),
                texts=islice(
                    LinearTransform(
                        initial=noise.fill(width=224, height=224), direction=direction
                    ),
                    0,
                    30 * speed * 2,
                    speed,
                ),
            )


if __name__ == "__main__":
    main(
        n_samples=30,
        direction=Direction.DOWN,
        labels="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        speed=1,
        mask_maker=_get_random_text_mask,
    )
