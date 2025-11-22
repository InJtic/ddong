from itertools import islice
from noise import BernoulliNoise
from transform import LinearTransform, Direction
from video import VideoGenerator


def main():
    width = 640
    height = 480
    noise = BernoulliNoise(prob=0.8)
    backgrounds = islice(
        LinearTransform(noise.fill(width, height), Direction.DOWN), 0, 1 * 3 * 30, 1
    )
    texts = islice(
        LinearTransform(noise.fill(width, height), Direction.UP), 0, 10 * 3 * 30, 10
    )

    vg = VideoGenerator(
        width=width,
        height=height,
        backgrounds=backgrounds,
        texts=texts,
        font_path="./resources/malgun.ttf",
        font_size=200,
        fps=60,
    )

    vg.save(text="똥", path="./ddong.avi")


if __name__ == "__main__":
    main()
