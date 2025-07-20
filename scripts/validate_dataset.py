import json

input_path = "scripts/fine_tune_chat_dataset.jsonl"
output_path = "scripts/fine_tune_chat_dataset_converted.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        item = json.loads(line.strip())
        prompt = item["prompt"]
        completion = item["completion"].strip()

        chat_format = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a health insurance claim decision assistant. Respond with a structured JSON object."
                },
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": completion
                }
            ]
        }

        outfile.write(json.dumps(chat_format) + "\n")

print("✅ Converted and saved to:", output_path)
