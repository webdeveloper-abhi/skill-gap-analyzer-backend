import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import fitz
from docx import Document
import tempfile
import re
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

NER_API = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"

SENTIMENT_API = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "skill_master.csv")

skill_df = pd.read_csv(CSV_PATH)

TECH_SKILLS = sorted(
    set(skill_df["skill"].dropna().astype(str).tolist())
)


def extract_text_from_pdf(path):

    text = ""

    doc = fitz.open(path)

    for page in doc:
        text += page.get_text()

    return text


def extract_text_from_docx(path):

    doc = Document(path)

    return "\n".join(
        [para.text for para in doc.paragraphs]
    )


def ner_request(text):

    payload = {
        "inputs": text[:700]
    }

    response = requests.post(
        NER_API,
        headers=HEADERS,
        json=payload,
        timeout=60
    )

    return response.json()


def sentiment_request(text):

    payload = {
        "inputs": text
    }

    response = requests.post(
        SENTIMENT_API,
        headers=HEADERS,
        json=payload,
        timeout=60
    )

    return response.json()


def extract_skills(text):

    found_skills = set()

    text_lower = text.lower()

    for skill in TECH_SKILLS:

        pattern = r'\b' + re.escape(skill.lower()) + r'\b'

        if re.search(pattern, text_lower):
            found_skills.add(skill)

    try:

        entities = ner_request(text)

        if isinstance(entities, list):

            for entity in entities:

                word = entity.get("word", "")

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

        strength = "MODERATE"

        if sentence:

            try:

                analysis = sentiment_request(sentence)

                if isinstance(analysis, list):

                    score = analysis[0]["score"]

                    if score > 0.7:
                        strength = "STRONG"

                    elif score > 0.4:
                        strength = "MODERATE"

                    else:
                        strength = "WEAK"

            except Exception as e:

                print("Sentiment Error:", e)

        results.append({
            "skill": skill,
            "strength": strength
        })

    return results


@app.route("/")
def home():

    return jsonify({
        "message": "Skill Gap Analyzer API Running Successfully"
    })


@app.route("/health")
def health():

    return jsonify({
        "status": "healthy"
    })


@app.route("/analyze_file", methods=["POST"])
def analyze_file():

    try:

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "Empty file"
            }), 400

        suffix = os.path.splitext(file.filename)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            file.save(temp_file.name)

            file_path = temp_file.name

        if suffix.lower() == ".pdf":

            text = extract_text_from_pdf(file_path)

        elif suffix.lower() == ".docx":

            text = extract_text_from_docx(file_path)

        else:

            return jsonify({
                "success": False,
                "error": "Unsupported file format"
            }), 400

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
        }), 500

    finally:

        try:

            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)

        except Exception as e:

            print("Cleanup Error:", e)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )