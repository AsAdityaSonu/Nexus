from transformers import pipeline
from trl import PPOTrainer, PPOConfig
import json

model = pipeline("text-generation", model="models/fine_tuned")

config = PPOConfig(batch_size=16)
ppo_trainer = PPOTrainer(model, config)

def train_self_learning_model(prompt, feedback):
    response = model(prompt)[0]["generated_text"]
    reward = 1 if feedback == "good" else -1
    ppo_trainer.step(prompt, response, reward)

def train_from_feedback():
    with open("data/chat_feedback.json", "r") as f:
        feedback_data = json.load(f)

    for entry in feedback_data:
        train_self_learning_model(entry["user"], entry["feedback"])

if __name__ == "__main__":
    train_from_feedback()
