from flask import Flask, request, session, render_template, url_for, redirect
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load model dan tokenizer
model_path = "./Model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/", methods=['GET', 'POST'])
def index():
    if "history" not in session:
        session["history"] = []
        
    if request.method == 'POST':
        question = request.form.get("question")
        if not question:
            return render_template("index.jinja", answer="Pertanyaannya kosong!")

        inputs = tokenizer.encode(question, return_tensors="pt", truncation=True).to(device)
        output_ids = model.generate(inputs, max_length=50)
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        session["history"].append({"question": question, "answer": answer})
        session.modified = True 

        return render_template("index.jinja", history=session.get("history", []))
    return render_template("index.jinja")

@app.route("/clear")
def clear():
    session.pop("history", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
