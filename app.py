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
import numpy as np
import imageio_ffmpeg

# Setup FFmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

# --- CONFIG ---
CONFIG_PATH = ".tutorial_sync_config.json"
OUTPUT_PATH = "voiceover_composition.mp4"

# --- LIGHT THEME UI ---
st.set_page_config(page_title="AI Voice-Over Studio", layout="wide")

st.markdown("""
<style>
    /* Light Theme Styling */
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
    
    h1, h2, h3 { color: #1a1a1a; font-weight: 700; }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Segment styling */
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
        # Try to find a valid flash or pro model
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        preferred = [m for m in models if "1.5-flash" in m]
        return preferred[0] if preferred else models[0]
    except Exception as e:
        return None

async def generate_voice_file(text, path, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)

def parse_ai_response(text):
    try:
        # Robust regex to find JSON even if AI adds extra conversational text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match: return []
        data = json.loads(match.group())
        return data.get("segments", [])
    except: return []

# --- CORE LOGIC ---
def assemble_composition(video_path, segments, voice):
    video = mp.VideoFileClip(video_path).without_audio()
    audio_clips = []
    temp_files = []

    for i, seg in enumerate(segments):
        if not seg['narration'].strip(): continue
        
        tmp_mp3 = f"temp_voice_{i}.mp3"
        asyncio.run(generate_voice_file(seg['narration'], tmp_mp3, voice))
        
        # Position audio layer on top of video at 'start' time
        clip = mp.AudioFileClip(tmp_mp3).set_start(seg['start'])
        audio_clips.append(clip)
        temp_files.append(tmp_mp3)

    if audio_clips:
        final_audio = mp.CompositeAudioClip(audio_clips).set_duration(video.duration)
        final_video = video.set_audio(final_audio)
    else:
        final_video = video

    final_video.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac", fps=24, logger=None)
    
    video.close()
    for f in temp_files: 
        if os.path.exists(f): os.remove(f)
    return OUTPUT_PATH

# --- MAIN APP ---
st.title("🎙️ AI Voice-Over Studio")
st.markdown("##### Transform silent footage into professional narrated content")

stored = load_config()
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password", value=stored.get("api_key", ""))
    voice = st.selectbox("Narrator Voice", [
        "en-US-JennyNeural", "en-US-GuyNeural", 
        "en-GB-SoniaNeural", "en-AU-NatashaNeural"
    ])
    if st.button("Save Settings"):
        save_config({"api_key": api_key})
        st.success("Settings saved!")

# 1. LAYER ONE: VIDEO IMPORT
st.markdown('<div class="layer-container video-label"><h3>Layer 1: Video Base</h3></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload silent video (MP4/MOV)", type=["mp4", "mov"], label_visibility="collapsed")

if uploaded_file:
    if st.session_state.video_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.getbuffer())
            st.session_state.video_path = tmp.name
        v = mp.VideoFileClip(st.session_state.video_path)
        st.session_state.video_duration = v.duration
        v.close()
    
    col_vid, col_info = st.columns([2, 1])
    with col_vid:
        st.video(st.session_state.video_path)
    with col_info:
        st.info(f"**File:** {uploaded_file.name}\n\n**Duration:** {st.session_state.video_duration:.2f} seconds")

# 2. LAYER TWO: VOICE OVER
st.markdown('<div class="layer-container audio-label"><h3>Layer 2: Voice-Over Segments</h3></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("✨ Auto-Generate", use_container_width=True):
        if not api_key:
            st.error("Please provide API Key in sidebar.")
        else:
            with st.spinner("AI analyzing video..."):
                target_model = get_valid_model(api_key)
                if not target_model:
                    st.error("Invalid API Key or Model access.")
                else:
                    model = genai.GenerativeModel(target_model)
                    video_file = genai.upload_file(path=st.session_state.video_path)
                    
                    # Wait for processing
                    while video_file.state.name == "PROCESSING":
                        time.sleep(2)
                        video_file = genai.get_file(video_file.name)
                    
                    prompt = f"Provide a high-quality narration script in JSON format. Match the visual actions. Output only JSON: {{'segments': [{{'start': 0.0, 'end': 5.0, 'narration': 'Text'}}]}}. Video duration: {st.session_state.video_duration}"
                    response = model.generate_content([video_file, prompt])
                    st.session_state.segments = parse_ai_response(response.text)
                    st.rerun()

if st.session_state.segments:
    for i, seg in enumerate(st.session_state.segments):
        with st.container():
            st.markdown(f'<div class="segment-box"><b>Clip #{i+1}</b>', unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns([1, 1, 4])
            st.session_state.segments[i]['start'] = sc1.number_input(f"Start Time", value=float(seg['start']), key=f"s_{i}", step=0.1)
            st.session_state.segments[i]['end'] = sc2.number_input(f"End Time", value=float(seg['end']), key=f"e_{i}", step=0.1)
            st.session_state.segments[i]['narration'] = sc3.text_area(f"Voice Script", value=seg['narration'], key=f"n_{i}", height=70)
            if st.button(f"Remove Clip {i+1}", key=f"del_{i}"):
                st.session_state.segments.pop(i)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    if st.button("➕ Add Manual Clip"):
        st.session_state.segments.append({"start": 0.0, "end": 2.0, "narration": ""})
        st.rerun()

# 3. EXPORT
st.divider()
if st.session_state.video_path and st.session_state.segments:
    if st.button("🚀 Render Final Composition", type="primary", use_container_width=True):
        with st.status("Compositing Layers...") as status:
            final_path = assemble_composition(st.session_state.video_path, st.session_state.segments, voice)
            status.update(label="Render Successful!", state="complete")
        
        st.success("Composition finished!")
        st.video(final_path)
        with open(final_path, "rb") as f:
            st.download_button("💾 Download Video", f, "final_voiceover.mp4", use_container_width=True)
