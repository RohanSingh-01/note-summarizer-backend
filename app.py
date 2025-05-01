from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load AI model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/summarize", methods=["POST"])
def summarize_text():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Empty input"}), 400

    try:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return jsonify({"summary": summary[0]['summary_text']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
