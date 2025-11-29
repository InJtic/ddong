import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model_path = "Qwen/Qwen3-VL-8B-Instruct"
video_path = "data/0/0.avi"
query = "Watch the video carefully. Which of the following options matches the text shown in the video?\n\n(A) 0\n(B) 1\n\nAnswer with the option letter directly."

# Load model (device_map="auto" lets transformers place layers on available GPUs)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_path)

# Read video and take the first frame as the image to describe
vr = VideoReader(video_path, ctx=cpu(0))
frames = vr.get_batch(range(len(vr))).asnumpy()

# Convert all frames to PIL images and include them in the message content
images = [Image.fromarray(f) for f in frames]
image_entries = [{"type": "image", "image": img} for img in images]

messages = [
    {
        "role": "user",
        "content": image_entries + [{"type": "text", "text": query}],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

# Move tensors to the model device(s)
for k, v in list(inputs.items()):
    if isinstance(v, torch.Tensor):
        inputs[k] = v.to(
            next(iter(model.device.values()))
            if isinstance(model.device, dict)
            else model.device
        )

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids) :]
    for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
