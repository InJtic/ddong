import argparse
import cv2
import os
import glob
from src import Direction


def images_to_video(
    speed: int,
    direction: Direction,
    label: str,
    sample_id: int,
    width: int = 224,
    height: int = 224,
    fps: int = 30,
    output_dir: str = "output",
):
    base_path = os.path.join("data", f"{speed}_{direction.name}", label)
    files = glob.glob(os.path.join(base_path, f"{sample_id}_*.png"))
    files.sort()

    size = (width, height)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(
        output_dir, f"{speed}_{direction.name}_{label}_sample{sample_id}.mp4"
    )

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)

    for file in files:
        img = cv2.imread(file)
        writer.write(img)

    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--speed", type=int, required=True)
    parser.add_argument(
        "--direction",
        type=str,
        required=True,
    )

    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--sample_id", type=int, required=True)

    args = parser.parse_args()

    images_to_video(
        speed=args.speed,
        direction=Direction[args.direction],
        label=args.label,
        sample_id=args.sample_id,
    )
