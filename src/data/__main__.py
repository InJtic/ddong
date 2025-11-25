from src.utils import load_default_data_settings
from src.data.generator import DataGenerator


def main():
    info = load_default_data_settings().build()
    data_generator = DataGenerator(info)

    for video in data_generator:
        video.save("./data/test-task")


if __name__ == "__main__":
    main()
