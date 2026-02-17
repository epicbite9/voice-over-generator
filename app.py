import asyncio
import json
import os
import re
import tempfile
import time
import uuid
import edge_tts
import google.generativeai as genai
import moviepy.editor as mp
import streamlit as st
import numpy as np
import imageio_ffmpeg

# Setup FFmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

# --- CONFIG ---
CONFIG_PATH = ".tutorial_sync_config.json"
OUTPUT_PATH = "voiceover_composition.mp4"

# --- UI THEME ---
st.set_page_config(page_title="AI Voice-Over Studio", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #212529; }
    .layer-container {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .video-label { border-left: 5px solid #007bff; }
    .audio-label { border-left: 5px solid #28a745; }
    .segment-box {
        background-color: #f1f3f5;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'segments' not in st.session_state: st.session_state.segments = []
if 'video_path' not in st.session_state: st.session_state.video_path = None
if 'video_duration' not in st.session_state: st.session_state.video_duration = 0.0

# --- HELPERS ---
def load_config():
    return json.load(open(CONFIG_PATH, "r")) if os.path.exists(CONFIG_PATH) else {}

def save_config(data):
    json.dump(data, open(CONFIG_PATH, "w"), indent=2)

def get_valid_model(api_key):
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        preferred = [m for m in models if "1.5-flash" in m]
        return preferred[0] if preferred else models[0]
    except: return None

async def generate_voice_file(text, path, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)

def parse_ai_response(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match: return []
        data = json.loads(match.group())
        return data.get("segments", [])
    except: return []

# --- FIX: ROBUST ASSEMBLY (No Echo) ---
def assemble_composition(video_path, segments, voice):
    # Load video and strip existing audio completely
    video = mp.VideoFileClip(video_path).without_audio()
    audio_clips = []
    temp_files = []

    # Sort segments by start time to prevent logical overlaps
    sorted_segments = sorted(segments, key=lambda x: x['start'])

    for i, seg in enumerate(sorted_segments):
        if not seg['narration'].strip(): continue
        
        # FIX: Use UUID to ensure every file is unique and doesn't "ghost" into the next render
        unique_id = str(uuid.uuid4())[:8]
        tmp_mp3 = os.path.join(tempfile.gettempdir(), f"voice_{unique_id}.mp3")
        
        asyncio.run(generate_voice_file(seg['narration'], tmp_mp3, voice))
        
        # Load clip
        a_clip = mp.AudioFileClip(tmp_mp3)
        # Position clip
        a_clip = a_clip.set_start(seg['start'])
        
        audio_clips.append(a_clip)
        temp_files.append(tmp_mp3)

    if audio_clips:
        # Create final composite audio track
        final_audio = mp.CompositeAudioClip(audio_clips).set_duration(video.duration)
        final_video = video.set_audio(final_audio)
    else:
        final_video = video

    # Write file
    final_video.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac", fps=24, logger=None)
    
    # CRITICAL: Close all handles to prevent memory echo
    final_video.close()
    video.close()
    for ac in audio_clips:
        ac.close()
    
    # Clean up files
    for f in temp_files:
        try:
            if os.path.exists(f): os.remove(f)
        except: pass
        
    return OUTPUT_PATH

# --- MAIN APP ---
st.title("🎙️ AI Voice-Over Studio")

stored = load_config()
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password", value=stored.get("api_key", ""))
    voice = st.selectbox("Narrator Voice", ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-AU-NatashaNeural"])
    if st.button("Save Settings"):
        save_config({"api_key": api_key})
        st.success("Saved!")

# LAYER 1: VIDEO
st.markdown('<div class="layer-container video-label"><h3>Layer 1: Video Base</h3></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov"], label_visibility="collapsed")

if uploaded_file:
    if st.session_state.video_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.getbuffer())
            st.session_state.video_path = tmp.name
        v = mp.VideoFileClip(st.session_state.video_path)
        st.session_state.video_duration = v.duration
        v.close()
    
    st.video(st.session_state.video_path)

# LAYER 2: AUDIO
st.markdown('<div class="layer-container audio-label"><h3>Layer 2: Voice-Over Segments</h3></div>', unsafe_allow_html=True)

if st.button("✨ Auto-Generate Script"):
    if api_key:
        with st.spinner("Analyzing..."):
            target_model = get_valid_model(api_key)
            model = genai.GenerativeModel(target_model)
            video_file = genai.upload_file(path=st.session_state.video_path)
            while video_file.state.name == "PROCESSING": time.sleep(2); video_file = genai.get_file(video_file.name)
            prompt = f"Return JSON: {{'segments': [{{'start': 0.0, 'end': 2.0, 'narration': '...'}}]}}. Video duration: {st.session_state.video_duration}"
            response = model.generate_content([video_file, prompt])
            st.session_state.segments = parse_ai_response(response.text)
            st.rerun()

if st.session_state.segments:
    for i, seg in enumerate(st.session_state.segments):
        with st.container():
            st.markdown(f'<div class="segment-box"><b>Clip #{i+1}</b>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 1, 4])
            st.session_state.segments[i]['start'] = c1.number_input("Start", value=float(seg['start']), key=f"s_{i}")
            st.session_state.segments[i]['end'] = c2.number_input("End", value=float(seg['end']), key=f"e_{i}")
            st.session_state.segments[i]['narration'] = c3.text_area("Script", value=seg['narration'], key=f"n_{i}")
            if st.button(f"Delete {i+1}", key=f"del_{i}"):
                st.session_state.segments.pop(i)
                st.rerun()

if st.button("➕ Add Manual Clip"):
    st.session_state.segments.append({"start": 0.0, "end": 2.0, "narration": ""})
    st.rerun()

# EXPORT
st.divider()
if st.session_state.video_path and st.session_state.segments:
    if st.button("🚀 Render Final Video", type="primary", use_container_width=True):
        with st.status("Rendering... (No Echo Fix Applied)") as status:
            final_path = assemble_composition(st.session_state.video_path, st.session_state.segments, voice)
            status.update(label="Complete!", state="complete")
        st.video(final_path)
        with open(final_path, "rb") as f:
            st.download_button("💾 Download", f, "output.mp4")
