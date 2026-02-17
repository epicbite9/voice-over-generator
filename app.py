import asyncio
import json
import os
import re
import subprocess
import tempfile
import time

import edge_tts
import google.generativeai as genai
import imageio_ffmpeg
import moviepy.editor as mp
import streamlit as st


CONFIG_PATH = ".tutorial_sync_config.json"
OUTPUT_PATH = "pro_demo.mp4"
GAP_EPSILON = 0.20
TARGET_WPS = 2.6
MIN_RATE_PERCENT = -15
MAX_RATE_PERCENT = 45


st.set_page_config(page_title="Tutorial Sync Studio", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1100px;}
.hero {
    background: linear-gradient(135deg, #0a2540 0%, #0f3b61 60%, #155987 100%);
    color: #f8fbff;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}
.hero h1 {margin: 0 0 0.35rem 0; font-size: 1.55rem;}
.hero p {margin: 0; opacity: 0.95;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>Tutorial Sync Studio</h1>
  <p>Generate narration and keep voice timing aligned with visual actions.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------- CONFIG ---------------- #

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
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
        preferred = [m for m in capable if "1.5" in m or "2.0" in m or "2.5" in m]
        return preferred[0] if preferred else (capable[0] if capable else None)
    except Exception:
        return None


# ---------------- AUDIO ---------------- #

async def generate_segment_audio_with_rate(text, output_path, voice, rate_percent):
    rate_percent = max(MIN_RATE_PERCENT, min(MAX_RATE_PERCENT, int(rate_percent)))
    rate = f"{rate_percent:+d}%"
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)


def measure_audio_duration(audio_path):
    audio = mp.AudioFileClip(audio_path)
    duration = float(audio.duration)
    audio.close()
    return duration


def build_timed_audio(text, audio_path, voice_choice, target_duration):
    rate_percent = 0
    for _ in range(3):
        asyncio.run(
            generate_segment_audio_with_rate(
                text, audio_path, voice_choice, rate_percent
            )
        )
        current_duration = measure_audio_duration(audio_path)
        diff = abs(current_duration - target_duration)
        if diff <= 0.22:
            break
        speed_factor = current_duration / max(0.2, target_duration)
        desired_adjust = int((speed_factor - 1.0) * 80)
        rate_percent += desired_adjust


# ---------------- SCRIPT PARSING ---------------- #

def parse_json_payload(raw_text):
    text = raw_text.strip()
    json_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    return json.loads(text)


def parse_segments_from_response(raw_text):
    try:
        payload = parse_json_payload(raw_text)
        segments = payload.get("segments", [])
        return [
            (float(s["start"]), float(s["end"]), s["narration"].strip())
            for s in segments
            if s.get("narration")
        ]
    except Exception:
        return []


# ---------------- VIDEO BUILD ---------------- #

def assemble_pro_video(original_video_path, script_data, voice_choice):
    video = mp.VideoFileClip(original_video_path)
    base_silent_video = video.without_audio()
    temp_audio_files = []
    voice_clips = []

    for i, (start_t, end_t, text) in enumerate(script_data):
        seg_path = f"seg_{i}.mp3"
        seg_duration = max(0.35, end_t - start_t)
        build_timed_audio(text, seg_path, voice_choice, seg_duration)
        temp_audio_files.append(seg_path)

        clip = mp.AudioFileClip(seg_path).set_start(start_t)
        voice_clips.append(clip)

    voice_track = mp.CompositeAudioClip(voice_clips).set_duration(video.duration)
    final_video = base_silent_video.set_audio(voice_track)

    final_video.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac", fps=24)

    final_video.close()
    video.close()

    for c in voice_clips:
        c.close()
    for p in temp_audio_files:
        os.remove(p)

    return OUTPUT_PATH


# ---------------- SIDEBAR (CLEAN) ---------------- #

stored = load_config()
default_key = stored.get("gemini_api_key", "")

with st.sidebar:
    st.subheader("Settings")

    key = st.text_input("Gemini API Key", type="password", value=default_key)
    save_key = st.checkbox("Remember API key", value=bool(default_key))

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

    if key and save_key and key != default_key:
        save_config({"gemini_api_key": key})
    elif not save_key:
        save_config({})

    model_name = get_best_model(key) if key else None
    if key and not model_name:
        st.warning("No compatible Gemini model found.")


# ---------------- MAIN UI ---------------- #

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


if uploaded_video and key and model_name:

    if st.button("Generate Professional Sync Demo", use_container_width=True):

        try:
            with st.status("Analyzing video and generating narration...") as status:

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(uploaded_video.getbuffer())
                tmp.close()

                video = mp.VideoFileClip(tmp.name)
                duration = float(video.duration)
                video.close()

                genai_file = genai.upload_file(path=tmp.name)
                while genai_file.state.name == "PROCESSING":
                    time.sleep(2)
                    genai_file = genai.get_file(genai_file.name)

                model = genai.GenerativeModel(model_name=model_name)

                prompt = f"""
Analyze this UI tutorial video and return JSON only.

Split into action-based segments.
Each segment must have:
- start (seconds)
- end (seconds)
- narration

Total video duration: {duration:.2f} seconds.

Return format:
{{
  "segments": [
    {{"start": 0.0, "end": 3.8, "narration": "..."}}
  ]
}}
"""

                response = model.generate_content([genai_file, prompt])
                script_data = parse_segments_from_response(response.text)

                if not script_data:
                    st.error("Could not generate script.")
                    st.stop()

                final_out = assemble_pro_video(
                    tmp.name,
                    script_data,
                    voice
                )

                status.update(label="Complete", state="complete")

            st.success("Professional demo created successfully.")
            st.video(final_out)

            with open(final_out, "rb") as f:
                st.download_button(
                    "Download Final Video",
                    f,
                    file_name="professional_sync_demo.mp4",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error: {e}")
