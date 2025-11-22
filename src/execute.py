import torch
import argparse
from src.convert import convert_inputs, get_images
from src.transform import Direction
from transformers import BatchFeature, Qwen2VLForConditionalGeneration, Qwen2VLProcessor


def get_answer(
    inputs: BatchFeature,
    model: Qwen2VLForConditionalGeneration,
    processor: Qwen2VLProcessor,
    max_new_tokens: int = 128,
) -> str:
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    raw_text = processor.decode(generated_ids_trimmed[0], skip_special_tokens=False)

    print("-" * 30)
    print(f"[DEBUG] Raw Tokens ID: {generated_ids_trimmed[0].tolist()}")
    print(f"[DEBUG] Raw Decoded Text: {repr(raw_text)}")
    print("-" * 30)

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]


def main():
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

    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    )
    processor = Qwen2VLProcessor.from_pretrained(model_id)

    prediction = get_answer(inputs=inputs, model=model, processor=processor)

    print(f"예측: {args.label} | 실제: {prediction}")


if __name__ == "__main__":
    main()
