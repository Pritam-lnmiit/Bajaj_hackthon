import openai
import os
from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY from .env

TRAIN_FILE = "scripts/fine_tune_dataset_prepared_train.jsonl"
VALID_FILE = "scripts/fine_tune_dataset_prepared_valid.jsonl"

# Upload files
train_file_id = openai.File.create(file=open(TRAIN_FILE, "rb"), purpose="fine-tune")["id"]
valid_file_id = openai.File.create(file=open(VALID_FILE, "rb"), purpose="fine-tune")["id"]

# Create fine-tuning job
response = openai.FineTuningJob.create(
    training_file=train_file_id,
    validation_file=valid_file_id,
    model="babbage-002"  # or "gpt-3.5-turbo", "davinci-002"
)

print(f"🎯 Fine-tune job started: {response['id']}")
