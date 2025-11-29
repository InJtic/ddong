import json
import ast


def format_options(options: list[str]):
    labels = "ABCD"

    formatted_str = ""
    for i, opt in enumerate(options):
        opt = opt.replace("'", "").replace('"', "").strip()
        formatted_str += f"({labels[i]}) {opt}\n"
    return formatted_str


def convert(item: dict):
    try:
        q_text = item["question"]
        lines = q_text.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("(") and line.strip().endswith(")"):
                option_line_idx = i
                break

        if option_line_idx == -1:
            return item

        raw_options = ast.literal_eval(lines[option_line_idx])
        new_options_str = format_options(raw_options)
        new_question = f"{lines[0]}\n\n{new_options_str}\n\n{lines[-1]}"

        item["question"] = new_question
        return item

    except Exception as e:
        print(f"Error processing item {item.get('video')}: {e}")
        return item


if __name__ == "__main__":
    processed_data = []

    with open("ddong_eval_data.jsonl", "r") as infile:
        for line in infile:
            item = json.loads(line)
            new_item = convert(item)
            processed_data.append(new_item)

    with open("processed_ddong_eval.jsonl", "w") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")
