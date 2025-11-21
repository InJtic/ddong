from noise import _NoiseArray
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from typing import Iterable, TypeAlias
import cv2

_Mask: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.bool]]


def _get_text_mask(
    width: int,
    height: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> _Mask:
    # 텍스트 drawer 얻기
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    bbox = draw.textbbox((0, 0), text, font=font)

    # 텍스트 중앙으로 정렬
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # 실제로 그리고 배열로 변환
    draw.text((x, y), text, font=font, fill=1)
    return np.array(mask).astype(np.bool)


class VideoGenerator:
    def __init__(
        self,
        width: int,
        height: int,
        backgrounds: Iterable[_NoiseArray],
        texts: Iterable[_NoiseArray],
        text: str,
        font_path: str | None = None,
        font_size: int = 150,
        length: int = 200,
        step: int = 1,
        fps: int = 30,
    ):
        """입력된 정보를 바탕으로 영상을 만듭니다.

        Args:
            width (int): 영상의 가로
            height (int): 영상의 세로
            backgrounds (Transform): 배경 전환기(Iterable)
            texts (Transform): 텍스트 전환기(Iterable)
            text (str): 텍스트
            font_path (str | None, optional): 텍스트 폰트. 지정되지 않으면 `ImageFont.load_default`를 사용합니다.
            font_size (int, optional): 텍스트 크기. 지정되지 않으면 `150`을 사용합니다.
            fps (int, optional): FPS. 기본값은 30입니다.
        """
        self.backgrounds = backgrounds
        self.texts = texts
        self.shape = (width, height)
        self.fps = fps

        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                font = ImageFont.load_default(font_size)
        else:
            font = ImageFont.load_default(font_size)

        self.mask = _get_text_mask(width, height, text, font)

    def save(self, path: str):
        """지정된 장소에 영상을 저장합니다.

        Args:
            path (str): _description_
        """
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, self.shape)

        # 텍스트/배경 가져와서 마스크 적용하고 합치기
        for text, background in zip(self.texts, self.backgrounds):
            frame = background.copy()
            frame[self.mask] = text[self.mask]
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(bgr)

        writer.release()
