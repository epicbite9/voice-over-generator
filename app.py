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


st.set_page_config(page_title="Tutorial Sync Studio", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1100px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #f7fafc 0%, #e9f0f8 100%);}
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


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
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
            m.name
            for m in models
            if "generateContent" in getattr(m, "supported_generation_methods", [])
            and "gemini" in m.name.lower()
        ]
        preferred = [m for m in capable if "1.5" in m or "2.0" in m or "2.5" in m]
        return preferred[0] if preferred else (capable[0] if capable else None)
    except Exception:
        return None


async def generate_segment_audio(text, output_path, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def can_read_first_frame(video_path):
    clip = None
    try:
        clip = mp.VideoFileClip(video_path)
        clip.get_frame(0)
        return True
    except Exception:
        return False
    finally:
        if clip is not None:
            clip.close()


def transcode_to_safe_mp4(input_path):
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_tmp.close()
    output_path = out_tmp.name

    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        output_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").strip()
        raise RuntimeError(f"FFmpeg transcode failed: {err[-500:]}")

    return output_path


def save_and_prepare_upload(uploaded_video):
    _, ext = os.path.splitext(uploaded_video.name or "")
    ext = ext.lower() if ext else ".mp4"
    raw_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    raw_tmp.write(uploaded_video.getbuffer())
    raw_tmp.close()

    if can_read_first_frame(raw_tmp.name):
        return raw_tmp.name, [raw_tmp.name]

    safe_path = transcode_to_safe_mp4(raw_tmp.name)
    if not can_read_first_frame(safe_path):
        raise RuntimeError("Uploaded video could not be decoded even after transcode.")
    return safe_path, [raw_tmp.name, safe_path]


def parse_segments_from_response(raw_text):
    text = raw_text.strip()
    json_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if json_match:
        text = json_match.group(1).strip()

    try:
        payload = json.loads(text)
        segments = payload.get("segments", payload) if isinstance(payload, (dict, list)) else []
        out = []
        for item in segments:
            if not isinstance(item, dict):
                continue
            start = float(item.get("start", item.get("start_sec", 0)))
            end = float(item.get("end", item.get("end_sec", 0)))
            narration = str(item.get("narration", item.get("text", ""))).strip()
            if narration:
                out.append((start, end, narration))
        return out
    except Exception:
        pass

    pattern = r"\[(\d+):(\d+(?:\.\d+)?)\s*-\s*(\d+):(\d+(?:\.\d+)?)\]\s*Text:\s*(.*)"
    matches = re.findall(pattern, raw_text)
    out = []
    for m in matches:
        start_sec = int(m[0]) * 60 + float(m[1])
        end_sec = int(m[2]) * 60 + float(m[3])
        out.append((start_sec, end_sec, m[4].strip()))
    return out


def normalize_segments(script_data, video_duration):
    cleaned = []
    for start, end, text in script_data:
        start = max(0.0, float(start))
        end = min(float(end), float(video_duration))
        if end <= start:
            continue
        txt = text.strip()
        if txt:
            cleaned.append((start, end, txt))

    cleaned.sort(key=lambda x: x[0])
    normalized = []
    for idx, (start, end, text) in enumerate(cleaned):
        next_start = cleaned[idx + 1][0] if idx + 1 < len(cleaned) else video_duration
        end = min(end, next_start)
        if end - start < 0.35:
            continue
        normalized.append((start, end, text))
    return normalized


def split_into_chunks(start, end, max_chunk=8.0):
    chunks = []
    cursor = start
    while cursor < end - 0.05:
        nxt = min(cursor + max_chunk, end)
        chunks.append((cursor, nxt))
        cursor = nxt
    return chunks


def coverage_ratio(script_data, video_duration):
    if video_duration <= 0:
        return 0.0
    total = sum(max(0.0, end - start) for start, end, _ in script_data)
    return min(1.0, total / video_duration)


def ensure_full_coverage(script_data, video_duration):
    if video_duration <= 0:
        return []

    base = normalize_segments(script_data, video_duration)
    if not base:
        base = [(0.0, video_duration, "In this step, follow the on-screen actions to continue the workflow.")]

    full = []
    cursor = 0.0
    for start, end, text in base:
        if start > cursor + GAP_EPSILON:
            for gs, ge in split_into_chunks(cursor, start):
                full.append((gs, ge, "Continue with the on-screen process and follow each visible action."))
        full.append((max(cursor, start), end, text))
        cursor = max(cursor, end)

    if cursor < video_duration - GAP_EPSILON:
        for gs, ge in split_into_chunks(cursor, video_duration):
            full.append((gs, ge, "Complete the next visible steps shown in this part of the tutorial."))

    adjusted = []
    for start, end, text in full:
        for cs, ce in split_into_chunks(start, end):
            if ce - cs >= 0.35:
                adjusted.append((cs, ce, text))
    return adjusted


def add_silence_padding(audio_clip, target_duration):
    pad = target_duration - audio_clip.duration
    if pad <= 0.05:
        return audio_clip
    silence = mp.AudioClip(lambda t: 0, duration=pad, fps=44100)
    return mp.concatenate_audioclips([audio_clip, silence])


def sync_clip_and_audio(clip, audio_clip):
    if audio_clip.duration > clip.duration + 0.08:
        freeze_duration = audio_clip.duration - clip.duration
        freeze_frame = clip.get_frame(max(0.0, clip.duration - 0.02))
        freeze_clip = mp.ImageClip(freeze_frame).set_duration(freeze_duration)
        clip = mp.concatenate_videoclips([clip, freeze_clip], method="compose")
    elif clip.duration > audio_clip.duration + 0.08:
        audio_clip = add_silence_padding(audio_clip, clip.duration)
    return clip, audio_clip


def assemble_pro_video(original_video_path, script_data, voice_choice):
    video = mp.VideoFileClip(original_video_path)
    segments = []
    temp_audio_files = []

    try:
        for i, entry in enumerate(script_data):
            start_t, end_t, text = entry
            if end_t <= start_t:
                continue

            clip = video.subclip(start_t, end_t)
            audio_seg_path = f"seg_{i}.mp3"
            asyncio.run(generate_segment_audio(text, audio_seg_path, voice_choice))
            temp_audio_files.append(audio_seg_path)
            audio_seg = mp.AudioFileClip(audio_seg_path)

            clip, audio_seg = sync_clip_and_audio(clip, audio_seg)
            clip = clip.set_audio(audio_seg)
            segments.append(clip)

        if not segments:
            raise ValueError("No valid segments were produced.")

        final_video = mp.concatenate_videoclips(segments, method="compose")
        final_video.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac", fps=24)
        final_video.close()
        return OUTPUT_PATH
    finally:
        for p in temp_audio_files:
            try:
                os.remove(p)
            except OSError:
                pass
        video.close()


stored = load_config()
default_key = stored.get("gemini_api_key", "")

with st.sidebar:
    st.subheader("Settings")
    key = st.text_input("Gemini API Key", type="password", value=default_key)
    save_key = st.checkbox("Remember API key on this computer", value=bool(default_key))
    voice = st.selectbox(
        "Narrator Voice",
        ["en-US-GuyNeural", "en-US-AvaNeural", "en-GB-SoniaNeural", "en-US-JennyNeural"],
    )

    if key and save_key and key != default_key:
        save_config({"gemini_api_key": key})
    elif (not save_key) and default_key:
        save_config({})

    model_name = get_best_model(key) if key else None
    if key and not model_name:
        st.warning("No compatible Gemini model found for this key.")
    elif model_name:
        st.caption(f"Using model: `{model_name}`")


col1, col2 = st.columns([1.5, 1])
with col1:
    uploaded_video = st.file_uploader("Upload Screen Recording", type=["mp4", "mov", "mkv"])
with col2:
    st.info(
        "Tips for better sync:\n"
        "- Use clear UI actions in the recording\n"
        "- Keep video resolution readable\n"
        "- Avoid long dead time between actions"
    )


if uploaded_video and key and model_name:
    if st.button("Generate Professional Sync Demo", use_container_width=True):
        temp_paths = []
        try:
            with st.status("Analyzing video and generating aligned narration...") as status:
                st.write("0/3 Preparing video file...")
                v_path, temp_paths = save_and_prepare_upload(uploaded_video)

                base_video = mp.VideoFileClip(v_path)
                duration = float(base_video.duration)
                base_video.close()

                st.write("1/3 Uploading video for scene analysis...")
                genai_file = genai.upload_file(path=v_path)
                while genai_file.state.name == "PROCESSING":
                    time.sleep(2)
                    genai_file = genai.get_file(genai_file.name)

                st.write("2/3 Creating structured timeline...")
                model = genai.GenerativeModel(model_name=model_name)
                prompt = f"""
Analyze this UI tutorial video and return JSON only.
Requirements:
- Split video into action-based segments.
- Each segment has: start (seconds), end (seconds), narration.
- Narration must fit naturally within each segment duration.
- Cover the video from start to finish with no overlap.
- Keep narration concise, professional, and tool-focused.
- Total video duration is approximately {duration:.2f} seconds.

Return format:
{{
  "segments": [
    {{"start": 0.0, "end": 3.8, "narration": "..." }}
  ]
}}
"""
                response = model.generate_content([genai_file, prompt])
                script_data = parse_segments_from_response(response.text)
                script_data = ensure_full_coverage(script_data, duration)
                cov = coverage_ratio(script_data, duration)
                if cov < 0.95:
                    script_data = ensure_full_coverage([], duration)

                if not script_data:
                    st.error("Could not parse action timestamps. Try a clearer recording.")
                    st.stop()

                st.write(f"3/3 Rendering final video from {len(script_data)} aligned segments...")
                final_out = assemble_pro_video(v_path, script_data, voice)
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
        finally:
            for p in temp_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
