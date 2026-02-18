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
OUTPUT_PATH = "final_voiceover_composition.mp4"

# --- UI THEME (Light) ---
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

# --- STATE ---
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

# --- STABLE ASSEMBLY ENGINE ---
def assemble_composition(video_path, segments, voice):
    # 1. Start fresh - Create a new temp directory for this specific render
    render_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join(tempfile.gettempdir(), f"render_{render_id}")
    os.makedirs(work_dir, exist_ok=True)
    
    # 2. Load video and strip audio
    video_clip = mp.VideoFileClip(video_path).without_audio()
    v_duration = video_clip.duration
    
    audio_clips = []
    
    # 3. Generate and prepare all audio segments
    for i, seg in enumerate(segments):
        if not seg['narration'].strip(): continue
        
        # Unique path inside work directory
        seg_mp3 = os.path.join(work_dir, f"seg_{i}.mp3")
        
        # Call edge-tts
        asyncio.run(generate_voice_file(seg['narration'], seg_mp3, voice))
        
        # Verify file exists and has content
        if os.path.exists(seg_mp3) and os.path.getsize(seg_mp3) > 0:
            a_clip = mp.AudioFileClip(seg_mp3)
            # Clip audio if it's longer than the video (safety)
            if seg['start'] < v_duration:
                a_clip = a_clip.set_start(seg['start'])
                audio_clips.append(a_clip)

    # 4. Create a single Composite Audio Track
    # We explicitly set duration to match video to prevent the "silence after 10s" bug
    if audio_clips:
        merged_audio = mp.CompositeAudioClip(audio_clips).set_duration(v_duration)
        final_video = video_clip.set_audio(merged_audio)
    else:
        final_video = video_clip

    # 5. Write to file using a unique temp name first to avoid locked-file echo
    temp_output = os.path.join(work_dir, "output_final.mp4")
    final_video.write_videofile(
        temp_output, 
        codec="libx264", 
        audio_codec="aac", 
        fps=24, 
        logger=None,
        temp_audiofile=os.path.join(work_dir, "temp_render_audio.m4a"),
        remove_temp=True
    )

    # 6. Final Cleanup of Memory
    final_video.close()
    video_clip.close()
    for ac in audio_clips:
        ac.close()
    
    # 7. Move result to permanent path
    import shutil
    shutil.copy(temp_output, OUTPUT_PATH)
    
    # 8. Clean up work directory
    try:
        shutil.rmtree(work_dir)
    except: pass
        
    return OUTPUT_PATH

# --- UI APP ---
st.title("🎙️ AI Voice-Over Studio")

stored = load_config()
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password", value=stored.get("api_key", ""))
    voice = st.selectbox("Narrator Voice", ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-AU-NatashaNeural"])
    if st.button("Save Settings"):
        save_config({"api_key": api_key})
        st.success("Config saved!")

# LAYER 1: VIDEO
st.markdown('<div class="layer-container video-label"><h3>Layer 1: Video Base</h3></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload video (silent)", type=["mp4", "mov"], label_visibility="collapsed")

if uploaded_file:
    if st.session_state.video_path is None or uploaded_file.name != st.session_state.get('last_file'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.getbuffer())
            st.session_state.video_path = tmp.name
            st.session_state['last_file'] = uploaded_file.name
        v = mp.VideoFileClip(st.session_state.video_path)
        st.session_state.video_duration = v.duration
        v.close()
    st.video(st.session_state.video_path)

# LAYER 2: AUDIO
st.markdown('<div class="layer-container audio-label"><h3>Layer 2: Voice-Over Segments</h3></div>', unsafe_allow_html=True)

if st.button("✨ Auto-Generate Script") and st.session_state.video_path:
    if api_key:
        with st.spinner("Analyzing Video..."):
            target_model = get_valid_model(api_key)
            model = genai.GenerativeModel(target_model)
            video_file = genai.upload_file(path=st.session_state.video_path)
            while video_file.state.name == "PROCESSING": time.sleep(2); video_file = genai.get_file(video_file.name)
            prompt = f"Return JSON script only. Total video duration is {st.session_state.video_duration}. Create segments across the whole video. Format: {{'segments': [{{'start': 0.0, 'end': 5.0, 'narration': '...'}}]}}"
            response = model.generate_content([video_file, prompt])
            st.session_state.segments = parse_ai_response(response.text)
            st.rerun()

# Display Segments
if st.session_state.segments:
    for i, seg in enumerate(st.session_state.segments):
        with st.container():
            st.markdown(f'<div class="segment-box"><b>Clip #{i+1}</b>', unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns([1, 1, 4])
            st.session_state.segments[i]['start'] = sc1.number_input(f"Start (s)", value=float(seg['start']), key=f"s_{i}")
            st.session_state.segments[i]['end'] = sc2.number_input(f"End (s)", value=float(seg['end']), key=f"e_{i}")
            st.session_state.segments[i]['narration'] = sc3.text_area(f"Voice Script", value=seg['narration'], key=f"n_{i}", height=70)
            if st.button(f"Delete Clip {i+1}", key=f"del_{i}"):
                st.session_state.segments.pop(i); st.rerun()

if st.button("➕ Add Manual Clip"):
    st.session_state.segments.append({"start": 0.0, "end": 2.0, "narration": ""})
    st.rerun()

# EXPORT
st.divider()
if st.session_state.video_path and st.session_state.segments:
    if st.button("🚀 Render Final Composition", type="primary", use_container_width=True):
        with st.status("Rendering Full Length Video (Fixing Echo & Silence)..."):
            final_path = assemble_composition(st.session_state.video_path, st.session_state.segments, voice)
            st.success("Render complete!")
        st.video(final_path)
        with open(final_path, "rb") as f:
            st.download_button("💾 Download Video", f, "final_tutorial.mp4", use_container_width=True)
