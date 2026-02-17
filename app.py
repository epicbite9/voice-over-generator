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
EDIT_OUTPUT_PATH = "edited_video.mp4"
GAP_EPSILON = 0.20
TARGET_WPS = 2.6
MIN_RATE_PERCENT = -15
MAX_RATE_PERCENT = 45


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


async def generate_segment_audio_with_rate(text, output_path, voice, rate_percent):
    rate_percent = max(MIN_RATE_PERCENT, min(MAX_RATE_PERCENT, int(rate_percent)))
    rate = f"{rate_percent:+d}%"
    communicate = edge_tts.Communicate(text, voice, rate=rate, volume="+0%")
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


def save_uploaded_file(uploaded_file, fallback_suffix):
    _, ext = os.path.splitext(uploaded_file.name or "")
    ext = ext.lower() if ext else fallback_suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


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


def parse_json_payload(raw_text):
    text = raw_text.strip()
    json_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if json_match:
        text = json_match.group(1).strip()
    return json.loads(text)


def parse_segments_from_response(raw_text):
    try:
        payload = parse_json_payload(raw_text)
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


def refine_script_for_professional_voice(model_name, script_data):
    if not script_data:
        return script_data

    compact = []
    for i, (start, end, text) in enumerate(script_data):
        duration = max(0.4, end - start)
        target_words = max(4, int(duration * TARGET_WPS))
        compact.append(
            {
                "index": i,
                "duration_sec": round(duration, 2),
                "target_words": target_words,
                "narration": text,
            }
        )

    prompt = """
Rewrite the narration lines to sound like a professional software tutorial voice-over.
Rules:
- Keep each line accurate to the original meaning. Do not invent UI elements or actions.
- Keep style clear, concise, and confident.
- Keep each line close to target_words so it can be spoken within duration_sec naturally.
- Return JSON only in this exact format:
{
  "segments": [
    {"index": 0, "narration": "..."}
  ]
}
"""
    try:
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content([prompt, json.dumps(compact)])
        payload = parse_json_payload(response.text)
        items = payload.get("segments", []) if isinstance(payload, dict) else []
        updates = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            narration = str(item.get("narration", "")).strip()
            if isinstance(idx, int) and narration:
                updates[idx] = narration
        if not updates:
            return script_data
        refined = []
        for i, (start, end, text) in enumerate(script_data):
            refined.append((start, end, updates.get(i, text)))
        return refined
    except Exception:
        return script_data


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


def measure_audio_duration(audio_path):
    audio = None
    try:
        audio = mp.AudioFileClip(audio_path)
        return float(audio.duration)
    finally:
        if audio is not None:
            audio.close()


def build_timed_audio(text, audio_path, voice_choice, target_duration):
    rate_percent = 0
    best_diff = float("inf")
    for _ in range(3):
        asyncio.run(generate_segment_audio_with_rate(text, audio_path, voice_choice, rate_percent))
        current_duration = measure_audio_duration(audio_path)
        diff = abs(current_duration - target_duration)
        if diff < best_diff:
            best_diff = diff
        if diff <= 0.22:
            break

        speed_factor = current_duration / max(0.2, target_duration)
        desired_adjust = int((speed_factor - 1.0) * 80)
        next_rate = max(MIN_RATE_PERCENT, min(MAX_RATE_PERCENT, rate_percent + desired_adjust))
        if next_rate == rate_percent:
            break
        rate_percent = next_rate


def add_silence_padding(audio_clip, target_duration):
    pad = target_duration - audio_clip.duration
    if pad <= 0.05:
        return audio_clip
    silence = mp.AudioClip(lambda t: 0, duration=pad, fps=44100)
    return mp.concatenate_audioclips([audio_clip, silence])


def apply_voice_cut(audio_clip, cut_start, cut_end, mode):
    total = float(audio_clip.duration)
    cut_start = max(0.0, min(float(cut_start), total))
    cut_end = max(cut_start, min(float(cut_end), total))
    if cut_end - cut_start < 0.05:
        return audio_clip

    before = audio_clip.subclip(0, cut_start) if cut_start > 0 else None
    after = audio_clip.subclip(cut_end, total) if cut_end < total else None

    if mode == "remove":
        parts = [p for p in [before, after] if p is not None]
        if not parts:
            return mp.AudioClip(lambda t: 0, duration=0.1, fps=44100)
        return mp.concatenate_audioclips(parts)

    silence = mp.AudioClip(lambda t: 0, duration=cut_end - cut_start, fps=44100)
    parts = [p for p in [before, silence, after] if p is not None]
    return mp.concatenate_audioclips(parts)


def apply_basic_edits(
    input_video_path,
    trim_start_sec,
    trim_end_sec,
    speed_factor,
    source_volume,
    bgm_path=None,
    bgm_volume=0.25,
):
    video = mp.VideoFileClip(input_video_path)
    bgm_clip = None
    edited = None

    try:
        start = max(0.0, float(trim_start_sec))
        end = float(trim_end_sec) if trim_end_sec > 0 else float(video.duration)
        end = min(end, float(video.duration))
        if end <= start + 0.2:
            raise ValueError("Trim range is too small. Increase end time.")

        edited = video.subclip(start, end)

        speed_factor = max(0.5, min(2.0, float(speed_factor)))
        if abs(speed_factor - 1.0) > 0.001:
            edited = edited.fx(mp.vfx.speedx, factor=speed_factor)

        final_audio = edited.audio.volumex(max(0.0, float(source_volume))) if edited.audio else None

        if bgm_path:
            bgm_clip = mp.AudioFileClip(bgm_path).volumex(max(0.0, float(bgm_volume)))
            if bgm_clip.duration < edited.duration:
                bgm_clip = mp.afx.audio_loop(bgm_clip, duration=edited.duration)
            else:
                bgm_clip = bgm_clip.subclip(0, edited.duration)

            final_audio = mp.CompositeAudioClip([final_audio, bgm_clip]) if final_audio else bgm_clip

        if final_audio:
            final_audio = final_audio.fx(mp.afx.audio_fadein, 0.25).fx(mp.afx.audio_fadeout, 0.35)
            edited = edited.set_audio(final_audio)

        output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_tmp.close()
        edited_path = output_tmp.name
        edited.write_videofile(edited_path, codec="libx264", audio_codec="aac", fps=24)
        return edited_path
    finally:
        try:
            if edited is not None:
                edited.close()
        except Exception:
            pass
        try:
            if bgm_clip is not None:
                bgm_clip.close()
        except Exception:
            pass
        try:
            video.close()
        except Exception:
            pass


def assemble_pro_video(
    original_video_path,
    script_data,
    voice_choice,
    voice_layer_shift_sec=0.0,
    voice_layer_volume=1.0,
    enable_voice_cut=False,
    voice_cut_start_sec=0.0,
    voice_cut_end_sec=0.0,
    voice_cut_mode="mute",
):
    video = mp.VideoFileClip(original_video_path)
    temp_audio_files = []
    voice_segment_clips = []

    try:
        base_silent_video = video.without_audio()

        for i, entry in enumerate(script_data):
            start_t, end_t, text = entry
            if end_t <= start_t:
                continue

            audio_seg_path = f"seg_{i}.mp3"
            seg_duration = max(0.35, end_t - start_t)
            build_timed_audio(text, audio_seg_path, voice_choice, seg_duration)
            temp_audio_files.append(audio_seg_path)
            audio_seg = mp.AudioFileClip(audio_seg_path).set_start(start_t).volumex(voice_layer_volume)
            voice_segment_clips.append(audio_seg)

        if not voice_segment_clips:
            raise ValueError("No valid segments were produced.")

        voice_track = mp.CompositeAudioClip(voice_segment_clips).set_duration(video.duration)

        if enable_voice_cut:
            voice_track = apply_voice_cut(voice_track, voice_cut_start_sec, voice_cut_end_sec, voice_cut_mode)

        voice_track = voice_track.set_start(voice_layer_shift_sec)
        if voice_track.duration < video.duration:
            voice_track = add_silence_padding(voice_track, video.duration)

        final_video = base_silent_video.set_audio(voice_track)
        final_video.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac", fps=24)
        final_video.close()
        return OUTPUT_PATH
    finally:
        for c in voice_segment_clips:
            try:
                c.close()
            except Exception:
                pass
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
    mode = st.radio(
        "Mode",
        ["AI Voiceover Sync", "Edit Only (CapCut Lite)"],
        help="Use AI narration sync, or only perform clean timeline edits.",
    )
    key = st.text_input("Gemini API Key", type="password", value=default_key)
    save_key = st.checkbox("Remember API key on this computer", value=bool(default_key))
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
    elif (not save_key) and default_key:
        save_config({})

    model_name = get_best_model(key) if key else None
    if key and not model_name:
        st.warning("No compatible Gemini model found for this key.")
    elif model_name:
        st.caption(f"Using model: `{model_name}`")

    st.markdown("---")
    st.subheader("Layer Mixer")
    st.caption("Layer 1: Main Video (silent base)")
    st.caption("Layer 2: Voiceover Script Audio")
    voice_layer_shift_sec = st.slider(
        "Move Voice Layer (sec)",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help="Move the full voice layer earlier/later on the timeline.",
    )
    voice_layer_volume = st.slider(
        "Voice Layer Volume",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.05,
    )
    enable_voice_cut = st.checkbox("Cut Voice Layer Range", value=False)
    voice_cut_start_sec = st.number_input("Voice Cut Start (sec)", min_value=0.0, value=0.0, step=0.5)
    voice_cut_end_sec = st.number_input("Voice Cut End (sec)", min_value=0.0, value=0.0, step=0.5)
    voice_cut_mode_label = st.selectbox(
        "Voice Cut Behavior",
        ["Mute only (keep timing)", "Remove and shift left"],
    )
    voice_cut_mode = "remove" if voice_cut_mode_label.startswith("Remove") else "mute"


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

st.subheader("Quick Edit Controls")
edit_c1, edit_c2, edit_c3, edit_c4 = st.columns(4)
with edit_c1:
    trim_start_sec = st.number_input("Trim Start (sec)", min_value=0.0, value=0.0, step=0.5)
with edit_c2:
    trim_end_sec = st.number_input(
        "Trim End (sec, 0 = full)", min_value=0.0, value=0.0, step=0.5
    )
with edit_c3:
    speed_factor = st.slider("Playback Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.05)
with edit_c4:
    source_volume = st.slider("Original Audio Volume", min_value=0.0, max_value=2.0, value=1.0, step=0.05)

bgm_file = st.file_uploader("Optional Background Music", type=["mp3", "wav", "m4a"])
bgm_volume = st.slider("BGM Volume", min_value=0.0, max_value=1.0, value=0.20, step=0.05)


if uploaded_video:
    run_label = "Render Edited Video" if mode == "Edit Only (CapCut Lite)" else "Generate Professional Sync Demo"
    can_run_ai = bool(key and model_name)
    if mode == "AI Voiceover Sync" and not can_run_ai:
        st.warning("Enter a valid Gemini API key to run AI voice-over mode.")

    if st.button(run_label, use_container_width=True, disabled=(mode == "AI Voiceover Sync" and not can_run_ai)):
        temp_paths = []
        try:
            status_text = "Applying edits and rendering..." if mode == "Edit Only (CapCut Lite)" else "Analyzing video and generating aligned narration..."
            with st.status(status_text) as status:
                st.write("0/3 Preparing video file...")
                v_path, temp_paths = save_and_prepare_upload(uploaded_video)
                bgm_path = None
                if bgm_file:
                    bgm_path = save_uploaded_file(bgm_file, ".mp3")
                    temp_paths.append(bgm_path)

                st.write("1/3 Applying timeline edits...")
                edited_video_path = apply_basic_edits(
                    input_video_path=v_path,
                    trim_start_sec=trim_start_sec,
                    trim_end_sec=trim_end_sec,
                    speed_factor=speed_factor,
                    source_volume=source_volume,
                    bgm_path=bgm_path,
                    bgm_volume=bgm_volume,
                )
                temp_paths.append(edited_video_path)

                if mode == "Edit Only (CapCut Lite)":
                    final_edit_path = EDIT_OUTPUT_PATH
                    with open(edited_video_path, "rb") as src, open(final_edit_path, "wb") as dst:
                        dst.write(src.read())
                    status.update(label="Complete", state="complete")
                    st.success("Edited video created successfully.")
                    st.video(final_edit_path)
                    with open(final_edit_path, "rb") as f:
                        st.download_button(
                            "Download Edited Video",
                            f,
                            file_name="edited_video.mp4",
                            use_container_width=True,
                        )
                    st.stop()

                base_video = mp.VideoFileClip(edited_video_path)
                duration = float(base_video.duration)
                base_video.close()

                st.write("2/3 Uploading video for scene analysis...")
                genai_file = genai.upload_file(path=edited_video_path)
                while genai_file.state.name == "PROCESSING":
                    time.sleep(2)
                    genai_file = genai.get_file(genai_file.name)

                st.write("3/3 Creating narration and rendering...")
                model = genai.GenerativeModel(model_name=model_name)
                prompt = f"""
Analyze this UI tutorial video and return JSON only.
Requirements:
- Split video into action-based segments.
- Each segment has: start (seconds), end (seconds), narration.
- Narration must fit naturally within each segment duration.
- Cover the video from start to finish with no overlap.
- Keep narration concise, professional, and tool-focused.
- Be visually grounded: describe only actions clearly visible on screen.
- Use an instructional voice-over tone, like a polished product demo.
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
                script_data = refine_script_for_professional_voice(model_name, script_data)
                cov = coverage_ratio(script_data, duration)
                if cov < 0.95:
                    script_data = ensure_full_coverage([], duration)

                if not script_data:
                    st.error("Could not parse action timestamps. Try a clearer recording.")
                    st.stop()

                final_out = assemble_pro_video(
                    edited_video_path,
                    script_data,
                    voice,
                    voice_layer_shift_sec=voice_layer_shift_sec,
                    voice_layer_volume=voice_layer_volume,
                    enable_voice_cut=enable_voice_cut,
                    voice_cut_start_sec=voice_cut_start_sec,
                    voice_cut_end_sec=voice_cut_end_sec,
                    voice_cut_mode=voice_cut_mode,
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
        finally:
            for p in temp_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
