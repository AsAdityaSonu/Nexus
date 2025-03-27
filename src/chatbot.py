from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from src.memory import ChatMemory

# Load model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Initialize memory system
memory = ChatMemory("data/chatHistory.json")

def chat_with_bot(prompt):
    # Retrieve past conversation history
    past_conversation = memory.get_last_n_conversations(5)
    full_prompt = past_conversation + f"\nUser: {prompt}\nChatbot: "

    # Encode and generate response
    inputs = tokenizer.encode(full_prompt + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=500, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)

    # Save conversation to memory
    memory.save_conversation(prompt, response)

    return response

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chat_with_bot(user_input)
        print("Chatbot:", response)
