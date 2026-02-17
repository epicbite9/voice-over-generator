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

# --- CONFIG ---
CONFIG_PATH = ".tutorial_sync_config.json"
OUTPUT_PATH = "final_voiceover_output.mp4"

st.set_page_config(page_title="Voice-Over Layer Studio", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e0e10; color: #efeff1; }
    .layer-card {
        background: #18181b;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3182ce;
        margin-bottom: 10px;
    }
    .video-layer { border-left: 5px solid #e53e3e; margin-bottom: 20px; }
    .audio-layer { border-left: 5px solid #38a169; }
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

async def generate_voice_file(text, path, voice, rate="+0%"):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(path)

def parse_ai_response(text):
    try:
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
        
        tmp_mp3 = f"layer_seg_{i}.mp3"
        # Generate high quality audio
        asyncio.run(generate_voice_file(seg['narration'], tmp_mp3, voice))
        
        # Create audio clip and set its position in the timeline
        clip = mp.AudioFileClip(tmp_mp3).set_start(seg['start'])
        
        # If segment 'end' is provided, we can optionally trim or speed up here
        # For simplicity, we align the start
        audio_clips.append(clip)
        temp_files.append(tmp_mp3)

    if audio_clips:
        final_audio = mp.CompositeAudioClip(audio_clips).set_duration(video.duration)
        final_video = video.set_audio(final_audio)
    else:
        final_video = video

    final_video.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac", fps=24)
    
    # Cleanup
    video.close()
    for f in temp_files: 
        if os.path.exists(f): os.remove(f)
    return OUTPUT_PATH

# --- UI ---
st.title("🎙️ Voice-Over Layer Editor")

stored = load_config()
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API Key", type="password", value=stored.get("api_key", ""))
    voice = st.selectbox("Narrator Voice", ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-AU-NatashaNeural"])
    if st.button("Save Settings"):
        save_config({"api_key": api_key})

# 1. LAYER ONE: VIDEO IMPORT
st.subheader("📺 Layer 1: Silent Video")
uploaded_file = st.file_uploader("Upload silent footage", type=["mp4", "mov"])

if uploaded_file:
    if st.session_state.video_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.getbuffer())
            st.session_state.video_path = tmp.name
        v = mp.VideoFileClip(st.session_state.video_path)
        st.session_state.video_duration = v.duration
        v.close()
    
    st.markdown(f'<div class="layer-card video-layer"><b>Main Video:</b> {uploaded_file.name} ({st.session_state.video_duration:.2f}s)</div>', unsafe_allow_html=True)

# 2. LAYER TWO: VOICE GENERATION / EDITING
st.subheader("🗣️ Layer 2: Voice-Over Segments")

col_tools, col_empty = st.columns([1, 2])
with col_tools:
    if st.button("🤖 AI Generate Script", use_container_width=True) and api_key:
        with st.spinner("Analyzing video..."):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Upload to Gemini
            video_file = genai.upload_file(path=st.session_state.video_path)
            while video_file.state.name == "PROCESSING": time.sleep(2); video_file = genai.get_file(video_file.name)
            
            prompt = f"Analyze this silent video and provide a high-quality narration script in JSON format. Return only: {{'segments': [{{'start': 0.0, 'end': 5.0, 'narration': '...'}}]}}. Total duration: {st.session_state.video_duration}"
            response = model.generate_content([video_file, prompt])
            st.session_state.segments = parse_ai_response(response.text)
            st.rerun()

    if st.button("➕ Add Manual Segment", use_container_width=True):
        st.session_state.segments.append({"start": 0.0, "end": 2.0, "narration": "New voice segment"})
        st.rerun()

# --- THE LAYER EDITOR ---
if st.session_state.segments:
    for i, seg in enumerate(st.session_state.segments):
        with st.container():
            st.markdown(f'<div class="layer-card audio-layer">Segment #{i+1}</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 1, 3])
            
            # Editable Layer Parameters
            st.session_state.segments[i]['start'] = c1.number_input(f"Start (s)", value=float(seg['start']), key=f"start_{i}", step=0.1)
            st.session_state.segments[i]['end'] = c2.number_input(f"End (s)", value=float(seg['end']), key=f"end_{i}", step=0.1)
            st.session_state.segments[i]['narration'] = c3.text_area(f"Narration Text", value=seg['narration'], key=f"txt_{i}", height=68)
            
            if st.button(f"🗑️ Delete Segment {i+1}", key=f"del_{i}"):
                st.session_state.segments.pop(i)
                st.rerun()

# 3. FINAL COMPOSITION
st.divider()
if st.session_state.video_path and st.session_state.segments:
    if st.button("🎬 Render Final Composition", type="primary", use_container_width=True):
        with st.status("Merging Layers...") as status:
            final_path = assemble_composition(st.session_state.video_path, st.session_state.segments, voice)
            status.update(label="Render Complete!", state="complete")
        
        st.video(final_path)
        with open(final_path, "rb") as f:
            st.download_button("💾 Download High-Quality Video", f, "output.mp4")
else:
    st.info("Upload a video and generate/add segments to begin.")
