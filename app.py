from flask import Flask, request, session, render_template, jsonify
import torch
from transformers import PegasusTokenizer,  PegasusForConditionalGeneration
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load model dan tokenizer
model_path = "./results/Model"
tokenizer = PegasusTokenizer.from_pretrained(model_path)
model = PegasusForConditionalGeneration.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/", methods=["GET", "POST"])
def index():
    history = session.get("history", [])

    if request.method == "POST":
        question = request.form.get("question")
        if not question:
            return render_template("index.jinja", answer="Pertanyaannya kosong!", history=history)

        inputs = tokenizer.encode(question, return_tensors="pt", truncation=True).to(device)
        output_ids = model.generate(inputs, max_length=50)
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        history.append({"question": question, "answer": answer})
        session["history"] = history 
        session.modified = True

        return render_template("index.jinja", history=history)

    return render_template("index.jinja", history=history)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Pertanyaan kosong"}), 400

    inputs = tokenizer.encode(question, return_tensors="pt", truncation=True).to(device)
    output_ids = model.generate(inputs, max_length=50)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    session.setdefault("history", [])
    session["history"].append({"question": question, "answer": answer})
    session.modified = True

    return jsonify({"answer": answer})


@app.route("/history")
def get_history():
    history = session.get("history", [])
    return jsonify({"history": history})


@app.route("/clear")
def clear():
    session.pop("history", None)
    return jsonify({"message": "Riwayat berhasil dihapus"})

if __name__ == "__main__":
    app.run(debug=True)
