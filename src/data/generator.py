from src.config import DataInfo
from dataclasses import dataclass
from src.utils import get_text_mask
from typing import Generator
from numpy.typing import NDArray
import numpy as np
import cv2
import os.path as osp
from pathlib import Path


@dataclass
class VideoData:
    index: int
    data: list[NDArray[np.uint8]]
    info: DataInfo

    def save(self, directory: str):
        path = osp.join(directory, f"{self.index}.avi")
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            filename=path,
            fourcc=fourcc,
            fps=self.info.video_info.fps,
            frameSize=(self.info.video_info.width, self.info.video_info.height),
            isColor=True,
        )
        for frame in self.data:
            if frame.ndim == 3 and frame.shape[2] == 3:
                writer.write(frame)
            else:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        writer.release()


class DataGenerator:
    def __init__(self, info: list[DataInfo]):
        self.info = info

    def __iter__(self) -> Generator[VideoData, None, None]:
        for index, info in enumerate(self.info):
            mask = get_text_mask(
                position=info.text_info.position,
                width=info.video_info.width,
                height=info.video_info.height,
                font=info.text_info.font,
                text=info.text_info.text,
            )
            frames: list[NDArray[np.uint8]] = []

            text_generator = info.text_info.transition.iter(info.text_info.initial)
            background_generator = info.background_info.transition.iter(
                info.background_info.initial
            )

            for text, background in zip(text_generator, background_generator):
                frame = background.copy()
                if frame.ndim == 3:
                    if text.ndim == 2:
                        text = cv2.cvtColor(text, cv2.COLOR_GRAY2BGR)
                else:
                    if text.ndim == 3:
                        text = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
                frame[mask] = text[mask]

                frames.append(frame)

            yield VideoData(index=index, data=frames, info=info)
