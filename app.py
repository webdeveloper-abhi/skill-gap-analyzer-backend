from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import pandas as pd
import fitz
from docx import Document
import tempfile
import os
import re
import torch

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading models...")

device = 0 if torch.cuda.is_available() else -1

ner = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
    device=-1
)

sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

print("Models loaded!")

skill_df = pd.read_csv("skill_master.csv")

TECH_SKILLS = sorted(set(skill_df["Skill"].dropna().astype(str).tolist()))
TECH_SKILLS_LOWER = [skill.lower() for skill in TECH_SKILLS]


def extract_text_from_pdf(path):
    text = ""
    doc = fitz.open(path)

    for page in doc:
        text += page.get_text()

    return text


def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_skills(text):
    found_skills = set()

    text_lower = text.lower()

    for skill in TECH_SKILLS:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'

        if re.search(pattern, text_lower):
            found_skills.add(skill)

    try:
        entities = ner(text[:3000])

        for entity in entities:
            word = entity['word']

            for skill in TECH_SKILLS:
                if word.lower() == skill.lower():
                    found_skills.add(skill)

    except Exception as e:
        print("NER Error:", e)

    return list(found_skills)


def find_best_sentence(text, skill):
    sentences = re.split(r'[.!?]', text)

    for sentence in sentences:
        if skill.lower() in sentence.lower():
            return sentence.strip()

    return ""


def analyze_skill_strength(text, skills):
    results = []

    for skill in skills:
        sentence = find_best_sentence(text, skill)

        if sentence:
            try:
                analysis = sentiment(sentence)[0]

                score = analysis['score']

                if score > 0.6:
                    strength = "STRONG"
                elif score > 0.3:
                    strength = "MODERATE"
                else:
                    strength = "WEAK"

            except:
                strength = "MODERATE"
        else:
            strength = "WEAK"

        results.append({
            "skill": skill,
            "strength": strength
        })

    return results


@app.route("/")
def home():
    return jsonify({
        "message": "Skill Gap Analyzer API Running"
    })


@app.route("/analyze_file", methods=["POST"])
def analyze_file():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        file.save(temp_file.name)
        file_path = temp_file.name

    try:

        if suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)

        elif suffix.lower() == ".docx":
            text = extract_text_from_docx(file_path)

        else:
            return jsonify({"error": "Unsupported file format"}), 400

        skills = extract_skills(text)

        analysis = analyze_skill_strength(text, skills)

        return jsonify({
            "success": True,
            "skills": analysis
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

    finally:
        os.remove(file_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)