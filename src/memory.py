import json
import os

class ChatMemory:
    def __init__(self, memory_file):
        self.memory_file = memory_file
        if not os.path.exists(memory_file):
            with open(memory_file, "w") as f:
                json.dump([], f)

    def save_conversation(self, user_input, chatbot_response):
        with open(self.memory_file, "r+") as f:
            chat_history = json.load(f)
            chat_history.append({"user": user_input, "chatbot": chatbot_response})
            f.seek(0)
            json.dump(chat_history, f, indent=4)

    def get_last_n_conversations(self, n=5):
        with open(self.memory_file, "r") as f:
            chat_history = json.load(f)
        last_n = chat_history[-n:]
        return "\n".join([f"User: {c['user']}\nChatbot: {c['chatbot']}" for c in last_n])
