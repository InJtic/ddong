import argparse
from PIL import Image
from transformers import BatchFeature, Qwen2VLProcessor
from src import Direction
import os
import glob

PROMPT = (
    "Step 1: Describe the shape and movement of the object hiding in the noise. "
    "(e.g. Does it have curves? Straight lines? Is it closed or open?) "
    "\n"
    "Step 2: Based on the description, identify the specific English alphabet. "
    "\n"
    "Final Answer Format: The letter is [X]."
)


def get_images(
    speed: int,
    direction: Direction,
    label: str,
    sample_id: int,
    width: int = 224,
    height: int = 224,
) -> list[Image.Image]:
    base_path = os.path.join("data", f"{speed}_{direction.name}", label)
    files = glob.glob(os.path.join(base_path, f"{sample_id}_*.png"))
    files.sort()

    return [Image.open(file).convert("RGB") for file in files]


def convert_inputs(
    frames: list[Image.Image],
    processor: Qwen2VLProcessor,
    prompt: str = PROMPT,
) -> BatchFeature:
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        conversation,
        template_kwargs={"add_generation_prompt": True},
    )

    inputs = processor(
        text=[prompt],
        videos=[frames],
        text_kwargs={"padding": True},
        common_kwargs={"return_tensors": "pt"},
    )

    return inputs


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

    inputs = convert_inputs(
        get_images(
            speed=args.speed,
            direction=Direction[args.direction],
            label=args.label,
            sample_id=args.sample_id,
        ),
        processor=Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct"),
    )

    print(f"Inputs keys: {inputs.keys()}")
    print(f"Video Shape: {inputs['pixel_values_videos'].shape}")
