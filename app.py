import os
import tempfile
import subprocess
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import whisper

app = Flask(__name__)
CORS(app)

# ─── CONFIG ───────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
whisper_model = whisper.load_model("base")

# ─── HEALTH CHECK ─────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "SAVEiT backend is running"})

# ─── MAIN PROCESS ENDPOINT ────────────────────────────
@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json()
        url = data.get("url", "").strip()
        keyword = data.get("keyword", "").strip()

        if not url or not keyword:
            return jsonify({"error": "URL and keyword are required"}), 400

        # ─── STEP 1: Download video using yt-dlp ──────
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            audio_path = os.path.join(tmpdir, "audio.mp3")

            try:
                subprocess.run([
                    "yt-dlp",
                    "--no-playlist",
                    "--max-filesize", "50m",
                    "-o", video_path,
                    url
                ], check=True, timeout=60, capture_output=True)
            except subprocess.CalledProcessError:
                # If download fails, skip transcription
                # Still generate description from keyword alone
                transcript = ""
            else:
                # ─── STEP 2: Extract audio ─────────────
                try:
                    subprocess.run([
                        "ffmpeg", "-i", video_path,
                        "-vn", "-acodec", "mp3",
                        audio_path, "-y"
                    ], check=True, timeout=30, capture_output=True)

                    # ─── STEP 3: Transcribe with Whisper ──
                    result = whisper_model.transcribe(
                        audio_path,
                        language=None,  # Auto detect Hindi/English/Mixed
                        task="transcribe"
                    )
                    transcript = result.get("text", "").strip()
                except Exception:
                    transcript = ""

        # ─── STEP 4: Generate smart description ───────
        prompt = f"""You are an AI assistant for SAVEiT app that helps users find their saved Instagram content later.

A user saved this Instagram content:
- URL: {url}
- Their keyword: {keyword}
- Transcript of the video (what was spoken): {transcript if transcript else "Not available"}

Your job is to generate a highly searchable description that will help the user find this content later using any search word they remember.

Please extract and include ALL of the following if present:
1. Main topic of the content
2. Any location, city, area, address mentioned (very important)
3. Any offer, discount, deal or price mentioned
4. Any time limit mentioned (valid till, offer ends, this week only etc)
5. Any food, product or service mentioned
6. Name of shop, restaurant, brand or person
7. Any contact number or address shown
8. Key details someone would search for weeks or months later

Write a detailed 5-6 sentence description entirely in English even if the video was in Hindi or Gujarati. Make every sentence highly searchable. Include all specific details. Do not write vague sentences.

Also at the end add a line starting with TAGS: and list 10-15 single word search tags separated by commas that someone might use to find this content."""

        response = gemini_model.generate_content(prompt)
        full_response = response.text.strip()

        # Split description and tags
        if "TAGS:" in full_response:
            parts = full_response.split("TAGS:")
            description = parts[0].strip()
            tags = [t.strip() for t in parts[1].split(",")]
        else:
            description = full_response
            tags = []

        return jsonify({
            "success": True,
            "transcript": transcript,
            "description": description,
            "tags": tags,
            "keyword": keyword,
            "url": url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── RUN ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
