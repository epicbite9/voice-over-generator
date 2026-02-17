import asyncio
import io
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
import numpy as np
import streamlit as st
from PIL import Image

# Ensure ffmpeg is found on Streamlit Cloud
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

CONFIG_PATH = ".tutorial_sync_config.json"
OUTPUT_PATH = "pro_demo.mp4"
GAP_EPSILON = 0.20
TARGET_WPS = 2.6
MIN_RATE_PERCENT = -15
MAX_RATE_PERCENT = 45

# Initialize session state
if 'segments' not in st.session_state:
    st.session_state.segments = []
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'video_duration' not in st.session_state:
    st.session_state.video_duration = 0.0
if 'bg_music' not in st.session_state:
    st.session_state.bg_music = None
if 'bg_music_volume' not in st.session_state:
    st.session_state.bg_music_volume = 0.15
if 'video_start' not in st.session_state:
    st.session_state.video_start = 0.0
if 'video_end' not in st.session_state:
    st.session_state.video_end = 0.0
if 'processed_audio_files' not in st.session_state:
    st.session_state.processed_audio_files = {}

st.set_page_config(page_title="Tutorial Sync Studio", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px;}
.hero {
    background: linear-gradient(135deg, #0a2540 0%, #0f3b61 60%, #155987 100%);
    color: #f8fbff;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}
.hero h1 {margin: 0; font-size: 1.55rem;}
.hero p {margin: 0; opacity: 0.95;}
.segment-item {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.segment-number {
    background: #3182ce;
    color: white;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: bold;
}
.timeline-container {
    background: #1a202c;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.timeline-bar {
    height: 40px;
    background: #2d3748;
    border-radius: 4px;
    position: relative;
    overflow: hidden;
}
.timeline-segment {
    position: absolute;
    height: 100%;
    background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.7rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="hero"><h1>Tutorial Sync Studio</h1><p>Complete video editing suite with AI narration.</p></div>', unsafe_allow_html=True)

# ==================== HELPERS ==================== #

def load_config():
    if not os.path.exists(CONFIG_PATH): return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_config(data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)

def get_best_model(api_key):
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        capable = [m.name for m in models if "generateContent" in getattr(m, "supported_generation_methods", []) and "gemini" in m.name.lower()]
        preferred = [m for m in capable if "1.5" in m or "2.0" in m]
        return preferred[0] if preferred else (capable[0] if capable else None)
    except: return None

# ==================== AUDIO ==================== #

async def generate_segment_audio_with_rate(text, output_path, voice, rate_percent, pitch_percent=0):
    rate = f"{int(rate_percent):+d}%"
    pitch = f"{int(pitch_percent):+d}%"
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_path)

def build_timed_audio(text, audio_path, voice_choice, target_duration, rate=0, pitch=0):
    rate_percent = rate
    for _ in range(3):
        asyncio.run(generate_segment_audio_with_rate(text, audio_path, voice_choice, rate_percent, pitch))
        audio = mp.AudioFileClip(audio_path)
        current_duration = float(audio.duration)
        audio.close()
        if abs(current_duration - target_duration) <= 0.22: break
        speed_factor = current_duration / max(0.2, target_duration)
        rate_percent += int((speed_factor - 1.0) * 80)

def generate_segment_audio(segment_idx, segment_data, voice, temp_dir):
    text = segment_data.get('text', '')
    if not text.strip(): return None
    seg_path = os.path.join(temp_dir, f"seg_{segment_idx}.mp3")
    seg_duration = max(0.35, segment_data.get('end', 0) - segment_data.get('start', 0))
    try:
        build_timed_audio(text, seg_path, voice, seg_duration, segment_data.get('rate', 0), segment_data.get('pitch', 0))
        return seg_path
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# ==================== PARSING ==================== #

def parse_json_payload(raw_text):
    text = raw_text.strip()
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match: text = json_match.group(1).strip()
        return json.loads(text)
    except:
        try: return json.loads(text)
        except: return {}

def parse_segments_from_response(raw_text, video_duration):
    payload = parse_json_payload(raw_text)
    segments = payload.get("segments", [])
    result = []
    for s in segments:
        start = float(s.get("start", 0))
        end = float(s.get("end", video_duration * 0.5))
        if end <= start: end = start + 2.0
        result.append({'start': min(start, video_duration), 'end': min(end, video_duration), 'text': s.get("narration", "").strip(), 'voice': 'en-US-JennyNeural', 'rate': 0, 'pitch': 0})
    return result

# ==================== VIDEO ==================== #

def assemble_pro_video(original_video_path, script_data, voice_choice, bg_music_path=None, bg_volume=0.15, video_start=0, video_end=None):
    video = mp.VideoFileClip(original_video_path)
    if video_end is None: video_end = video.duration
    video = video.subclip(video_start, video_end)
    base_video = video.without_audio()
    
    temp_audio_files = []
    voice_clips = []

    for i, segment in enumerate(script_data):
        if not segment.get('text', '').strip(): continue
        start_t = segment.get('start', 0) - video_start
        end_t = segment.get('end', 0) - video_start
        seg_duration = max(0.35, end_t - start_t)
        seg_path = f"export_seg_{i}.mp3"
        temp_audio_files.append(seg_path)
        build_timed_audio(segment['text'], seg_path, voice_choice, seg_duration, segment.get('rate', 0), segment.get('pitch', 0))
        clip = mp.AudioFileClip(seg_path).set_start(max(0, start_t))
        voice_clips.append(clip)

    voice_track = mp.CompositeAudioClip(voice_clips).set_duration(video.duration) if voice_clips else None
    final_audio = voice_track

    if bg_music_path and os.path.exists(bg_music_path):
        bg = mp.AudioFileClip(bg_music_path)
        if bg.duration < video.duration:
            bg = mp.concatenate_audioclips([bg] * int(np.ceil(video.duration / bg.duration)))
        bg = bg.subclip(0, video.duration).volumex(bg_volume)
        final_audio = mp.CompositeAudioClip([bg, voice_track]) if voice_track else bg

    final_video = base_video.set_audio(final_audio) if final_audio else base_video
    final_video.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac", fps=24, logger=None)
    
    # Cleanup
    final_video.close(); video.close()
    for p in temp_audio_files: 
        try: os.remove(p)
        except: pass
    return OUTPUT_PATH

# ==================== MAIN UI ==================== #

stored = load_config()
with st.sidebar:
    st.subheader("Settings")
    key = st.text_input("Gemini API Key", type="password", value=stored.get("gemini_api_key", ""))
    save_key = st.checkbox("Remember API key", value=bool(stored.get("gemini_api_key", "")))
    voice_options = ["en-US-AndrewMultilingualNeural", "en-US-AvaNeural", "en-US-GuyNeural", "en-US-JennyNeural", "en-GB-SoniaNeural"]
    default_voice = st.selectbox("Default Voice", voice_options, index=3)
    if key and save_key: save_config({"gemini_api_key": key, "default_voice": default_voice})
    model_name = get_best_model(key) if key else None
    st.divider()
    if st.button("Clear Project", use_container_width=True):
        st.session_state.segments = []; st.session_state.video_path = None; st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["Video", "Segments", "Audio", "Export"])

# TAB 1: VIDEO
with tab1:
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        if st.session_state.video_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.getbuffer())
                st.session_state.video_path = tmp.name
            v = mp.VideoFileClip(st.session_state.video_path)
            st.session_state.video_duration = float(v.duration)
            st.session_state.video_end = st.session_state.video_duration
            v.close()
        
        st.video(st.session_state.video_path)
        
        # Robust Slider Handling to prevent StreamlitAPIException
        max_dur = max(0.1, float(st.session_state.video_duration))
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.video_start = st.slider("Start Time (s)", 0.0, max_dur, min(st.session_state.video_start, max_dur), 0.1, key="trim_start")
        with col2:
            st.session_state.video_end = st.slider("End Time (s)", 0.0, max_dur, min(st.session_state.video_end, max_dur), 0.1, key="trim_end")
        st.info(f"Trimmed Duration: {st.session_state.video_end - st.session_state.video_start:.1f}s")

# TAB 2: SEGMENTS
with tab2:
    if st.session_state.video_path:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c2:
            if st.button("Add Segment", use_container_width=True):
                last = st.session_state.segments[-1]['end'] if st.session_state.segments else 0
                st.session_state.segments.append({'start': last, 'end': min(last+2, st.session_state.video_duration), 'text': '', 'voice': default_voice, 'rate': 0, 'pitch': 0})
                st.rerun()
        with c3:
            if st.button("AI Generate", use_container_width=True, type="primary") and key:
                with st.status("AI Generating..."):
                    genai_file = genai.upload_file(path=st.session_state.video_path)
                    while genai_file.state.name == "PROCESSING": time.sleep(2); genai_file = genai.get_file(genai_file.name)
                    prompt = f"Analyze video and return JSON with 'segments' list (start, end, narration). Total duration: {st.session_state.video_duration}s."
                    response = genai.GenerativeModel(model_name=model_name).generate_content([genai_file, prompt])
                    st.session_state.segments = parse_segments_from_response(response.text, st.session_state.video_duration)
                st.rerun()

        # Timeline
        if st.session_state.segments:
            tl = '<div class="timeline-container"><div class="timeline-bar">'
            total = max(0.1, st.session_state.video_end - st.session_state.video_start)
            for i, s in enumerate(st.session_state.segments):
                left = ((s['start'] - st.session_state.video_start) / total) * 100
                w = ((s['end'] - s['start']) / total) * 100
                tl += f'<div class="timeline-segment" style="left:{max(0, left)}%; width:{max(2, w)}%;">{i+1}</div>'
            st.markdown(tl + '</div></div>', unsafe_allow_html=True)

            for i, seg in enumerate(st.session_state.segments):
                with st.expander(f"Segment {i+1}: {seg['text'][:30]}...", expanded=True):
                    seg['text'] = st.text_area("Text", seg['text'], key=f"t_{i}")
                    ca, cb = st.columns(2)
                    max_d = max(0.1, st.session_state.video_duration)
                    seg['start'] = ca.slider("Start", 0.0, max_d, float(seg['start']), 0.1, key=f"s_{i}")
                    seg['end'] = cb.slider("End", 0.0, max_d, float(seg['end']), 0.1, key=f"e_{i}")
                    v1, v2, v3 = st.columns(3)
                    seg['voice'] = v1.selectbox("Voice", voice_options, index=voice_options.index(seg['voice']) if seg['voice'] in voice_options else 3, key=f"v_{i}")
                    seg['rate'] = v2.slider("Speed", -15, 45, int(seg['rate']), key=f"r_{i}")
                    if st.button(f"Delete {i+1}", key=f"del_{i}"):
                        st.session_state.segments.pop(i); st.rerun()

# TAB 3: AUDIO
with tab3:
    bg_up = st.file_uploader("Background Music", type=["mp3", "wav"])
    if bg_up:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(bg_up.getbuffer()); st.session_state.bg_music = tmp.name
    if st.session_state.bg_music:
        st.audio(st.session_state.bg_music)
        st.session_state.bg_music_volume = st.slider("Volume", 0.0, 1.0, st.session_state.bg_music_volume)

# TAB 4: EXPORT
with tab4:
    if st.session_state.video_path and st.session_state.segments:
        if st.button("Generate Final Video", type="primary", use_container_width=True):
            out = assemble_pro_video(st.session_state.video_path, st.session_state.segments, default_voice, st.session_state.bg_music, st.session_state.bg_music_volume, st.session_state.video_start, st.session_state.video_end)
            st.video(out)
            with open(out, "rb") as f: st.download_button("Download", f, "final_video.mp4")
    else: st.info("Upload video and add segments to export.")
