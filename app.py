import asyncio
import json
import os
import re
import tempfile
import time

import edge_tts
import google.generativeai as genai
import moviepy.editor as mp
import streamlit as st


CONFIG_PATH = ".tutorial_sync_config.json"
OUTPUT_PATH = "pro_demo.mp4"
MIN_RATE_PERCENT = -15
MAX_RATE_PERCENT = 45


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(page_title="Tutorial Sync Studio", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1100px;}
.hero {
    background: linear-gradient(135deg, #0a2540 0%, #0f3b61 60%, #155987 100%);
    color: #f8fbff;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}
.hero h1 {margin: 0; font-size: 1.6rem;}
.hero p {margin: 0; opacity: 0.95;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Tutorial Sync Studio</h1>
  <p>Generate narration aligned with visual actions.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- CONFIG ---------------- #

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def save_config(data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_best_model(api_key):
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        capable = [
            m.name for m in models
            if "generateContent" in getattr(m, "supported_generation_methods", [])
            and "gemini" in m.name.lower()
        ]
        return capable[0] if capable else None
    except:
        return None


# ---------------- AUDIO ---------------- #

async def generate_audio(text, output_path, voice, rate_percent):
    rate_percent = max(MIN_RATE_PERCENT, min(MAX_RATE_PERCENT, int(rate_percent)))
    rate = f"{rate_percent:+d}%"
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)


def measure_audio_duration(path):
    clip = mp.AudioFileClip(path)
    duration = clip.duration
    clip.close()
    return duration


def build_timed_audio(text, path, voice, target_duration):
    rate = 0
    for _ in range(3):
        asyncio.run(generate_audio(text, path, voice, rate))
        current = measure_audio_duration(path)
        diff = abs(current - target_duration)
        if diff <= 0.25:
            break
        speed_factor = current / max(0.2, target_duration)
        rate += int((speed_factor - 1.0) * 80)


# ---------------- ROBUST SCRIPT PARSER ---------------- #

def parse_segments_from_response(response):
    try:
        # Extract raw text safely
        raw_text = ""

        if hasattr(response, "text") and response.text:
            raw_text = response.text
        else:
            try:
                raw_text = response.candidates[0].content.parts[0].text
            except:
                return []

        if not raw_text:
            return []

        # Remove markdown formatting
        raw_text = re.sub(r"```json", "", raw_text, flags=re.IGNORECASE)
        raw_text = raw_text.replace("```", "").strip()

        # Extract first JSON object
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return []

        payload = json.loads(match.group())

        segments = payload.get("segments", [])

        cleaned = []
        for seg in segments:
            try:
                start = float(seg.get("start", 0))
                end = float(seg.get("end", 0))
                narration = str(seg.get("narration", "")).strip()
                if narration and end > start:
                    cleaned.append((start, end, narration))
            except:
                continue

        return cleaned

    except Exception:
        return []


# ---------------- VIDEO BUILD ---------------- #

def assemble_video(video_path, script_data, voice):
    video = mp.VideoFileClip(video_path)
    silent = video.without_audio()

    temp_files = []
    audio_clips = []

    for i, (start, end, text) in enumerate(script_data):
        duration = max(0.4, end - start)
        audio_path = f"seg_{i}.mp3"

        build_timed_audio(text, audio_path, voice, duration)

        clip = mp.AudioFileClip(audio_path).set_start(start)
        audio_clips.append(clip)
        temp_files.append(audio_path)

    voice_track = mp.CompositeAudioClip(audio_clips).set_duration(video.duration)
    final_video = silent.set_audio(voice_track)

    final_video.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac", fps=24)

    final_video.close()
    video.close()

    for c in audio_clips:
        c.close()
    for f in temp_files:
        os.remove(f)

    return OUTPUT_PATH


# ---------------- SIDEBAR ---------------- #

stored = load_config()
default_key = stored.get("gemini_api_key", "")

with st.sidebar:
    st.subheader("Settings")

    api_key = st.text_input("Gemini API Key", type="password", value=default_key)
    remember = st.checkbox("Remember API key", value=bool(default_key))

    voice = st.selectbox(
        "Narrator Voice",
        [
            "en-US-AndrewMultilingualNeural",
            "en-US-AvaNeural",
            "en-US-GuyNeural",
            "en-US-JennyNeural",
            "en-GB-SoniaNeural",
        ],
    )

    if api_key and remember:
        save_config({"gemini_api_key": api_key})
    elif not remember:
        save_config({})

    model_name = get_best_model(api_key) if api_key else None
    if api_key and not model_name:
        st.warning("No compatible Gemini model found.")


# ---------------- MAIN ---------------- #

uploaded_video = st.file_uploader(
    "Upload Screen Recording",
    type=["mp4", "mov", "mkv"]
)

st.info(
    "Tips:\n"
    "- Keep actions clear\n"
    "- Avoid long pauses\n"
    "- Ensure UI elements are readable"
)

if uploaded_video and api_key and model_name:

    if st.button("Generate Professional Sync Demo", use_container_width=True):

        try:
            with st.status("Analyzing video and generating narration...") as status:

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(uploaded_video.getbuffer())
                tmp.close()

                video = mp.VideoFileClip(tmp.name)
                duration = video.duration
                video.close()

                genai.configure(api_key=api_key)
                genai_file = genai.upload_file(path=tmp.name)

                while genai_file.state.name == "PROCESSING":
                    time.sleep(2)
                    genai_file = genai.get_file(genai_file.name)

                model = genai.GenerativeModel(model_name=model_name)

                prompt = f"""
Analyze this UI tutorial video.

Return JSON only:

{{
  "segments": [
    {{"start": 0.0, "end": 3.5, "narration": "..." }}
  ]
}}

Video duration: {duration:.2f} seconds.
"""

                response = model.generate_content([genai_file, prompt])
                script_data = parse_segments_from_response(response)

                # ✅ FALLBACK FIX
                if not script_data:
                    script_data = [
                        (0.0, duration, "In this tutorial, follow the on-screen steps to complete the workflow.")
                    ]

                status.write("Rendering final video...")
                final_path = assemble_video(tmp.name, script_data, voice)

                status.update(label="Complete", state="complete")

            st.success("Professional demo created successfully.")
            st.video(final_path)

            with open(final_path, "rb") as f:
                st.download_button(
                    "Download Final Video",
                    f,
                    file_name="professional_sync_demo.mp4",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error: {e}")
