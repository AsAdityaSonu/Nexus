from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import json
import torch

MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def load_conversation_history(file_path):
    with open(file_path, "r") as f:
        chat_history = json.load(f)
    
    # Format data for fine-tuning
    dataset = [{"text": f"User: {c['user']}\nChatbot: {c['chatbot']}"} for c in chat_history]
    return Dataset.from_list(dataset)

def fine_tune_model():
    dataset = load_conversation_history("data/chat_history.json")

    training_args = TrainingArguments(
        output_dir="models/fine_tuned",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained("models/fine_tuned")

if __name__ == "__main__":
    fine_tune_model()
