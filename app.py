from flask import Flask, request, jsonify
from src.chatbot import chat_with_bot

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = chat_with_bot(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5000)
