import asyncio
import json
import os
import re
import tempfile
import time
import uuid
import shutil
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
OUTPUT_PATH = "tutor_voiceover_final.mp4"

# --- UI THEME ---
st.set_page_config(page_title="AI Tutorial Studio", layout="wide")
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

# --- TUTOR ENGINE ---
def assemble_composition(video_path, segments, voice):
    render_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join(tempfile.gettempdir(), f"tutor_{render_id}")
    os.makedirs(work_dir, exist_ok=True)
    
    video_clip = mp.VideoFileClip(video_path).without_audio()
    v_dur = video_clip.duration
    
    sorted_segs = sorted(segments, key=lambda x: x['start'])
    final_audio_clips = []
    last_audio_end_time = 0.0 
    
    for i, seg in enumerate(sorted_segs):
        if not seg['narration'].strip(): continue
        
        seg_mp3 = os.path.join(work_dir, f"seg_{i}.mp3")
        asyncio.run(generate_voice_file(seg['narration'], seg_mp3, voice))
        
        if os.path.exists(seg_mp3) and os.path.getsize(seg_mp3) > 0:
            a_clip = mp.AudioFileClip(seg_mp3)
            start_time = float(seg['start'])
            
            # Anti-Overlap: Ensure segments don't talk over each other
            if start_time < last_audio_end_time:
                start_time = last_audio_end_time + 0.15
            
            if start_time < v_dur:
                a_clip = a_clip.set_start(start_time)
                final_audio_clips.append(a_clip)
                last_audio_end_time = start_time + a_clip.duration
            else:
                a_clip.close()

    if final_audio_clips:
        merged_audio = mp.CompositeAudioClip(final_audio_clips).set_duration(v_dur)
        final_video = video_clip.set_audio(merged_audio)
    else:
        final_video = video_clip

    temp_out = os.path.join(work_dir, "final.mp4")
    final_video.write_videofile(
        temp_out, codec="libx264", audio_codec="aac", fps=24, logger=None,
        temp_audiofile=os.path.join(work_dir, "audio.m4a"), remove_temp=True
    )

    final_video.close()
    video_clip.close()
    for ac in final_audio_clips: ac.close()
    
    shutil.copy(temp_out, OUTPUT_PATH)
    shutil.rmtree(work_dir, ignore_errors=True)
    return OUTPUT_PATH

# --- UI ---
st.title("🎓 AI Tutor Narrator")
st.markdown("##### The AI acts as a teacher, giving direct instructions to your visitors.")

stored = load_config()
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password", value=stored.get("api_key", ""))
    voice = st.selectbox("Narrator Voice", ["en-US-AndrewMultilingualNeural", "en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"])
    if st.button("Save Settings"): save_config({"api_key": api_key})

# LAYER 1
st.markdown('<div class="layer-container video-label"><h3>Layer 1: Tutorial Video</h3></div>', unsafe_allow_html=True)
up = st.file_uploader("Import silent video", type=["mp4", "mov"], label_visibility="collapsed")
if up:
    if st.session_state.video_path is None or up.name != st.session_state.get('last_fn'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
            t.write(up.getbuffer()); st.session_state.video_path = t.name
            st.session_state['last_fn'] = up.name
        v = mp.VideoFileClip(st.session_state.video_path)
        st.session_state.video_duration = v.duration
        v.close()
    st.video(st.session_state.video_path)

# LAYER 2
st.markdown('<div class="layer-container audio-label"><h3>Layer 2: Instructional Voice Layer</h3></div>', unsafe_allow_html=True)
if st.button("✨ Generate Tutorial Instructions") and st.session_state.video_path:
    if api_key:
        with st.spinner("AI is crafting the tutorial..."):
            m = genai.GenerativeModel(get_valid_model(api_key))
            vf = genai.upload_file(path=st.session_state.video_path)
            while vf.state.name == "PROCESSING": time.sleep(2); vf = genai.get_file(vf.name)
            
            # --- THE TUTORIAL PERSONA PROMPT ---
            prompt = f"""
            You are a professional TEACHER and TUTOR. 
            Analyze the video and write a narration script to guide a VISITOR through the steps.
            
            STRICT PERSONA RULES:
            1. Use the SECOND-PERSON ("You") and IMPERATIVE MOOD (Commands).
            2. TEACH the viewer. Say "Click on...", "Navigate to...", "You can see that...".
            3. DO NOT describe your own actions. Never say "I am doing X" or "I click Y".
            4. Use "Next, you want to..." or "Go ahead and click...".
            5. Address the visitor as "You". Example: "First, navigate to your WordPress dashboard."
            6. Total duration: {st.session_state.video_duration} seconds.
            
            Example of GOOD style:
            "In this video, I'll show you how you can install the plugin. First, go to Plugins and then click Add New."
            
            Example of BAD style:
            "In this video, I'm going to show you how I install the plugin. First, I go to Plugins..."
            
            Return JSON only:
            {{ "segments": [ {{ "start": 0.0, "end": 5.0, "narration": "..." }} ] }}
            """
            
            res = m.generate_content([vf, prompt])
            st.session_state.segments = parse_ai_response(res.text)
            st.rerun()

if st.session_state.segments:
    for i, seg in enumerate(st.session_state.segments):
        with st.container():
            st.markdown(f'<div class="segment-box"><b>Instruction {i+1}</b>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 1, 4])
            st.session_state.segments[i]['start'] = c1.number_input("Time(s)", value=float(seg['start']), key=f"s_{i}")
            st.session_state.segments[i]['end'] = c2.number_input("To(s)", value=float(seg['end']), key=f"e_{i}")
            st.session_state.segments[i]['narration'] = c3.text_area("Tutor Script", value=seg['narration'], key=f"n_{i}")
            if st.button(f"Remove {i+1}", key=f"del_{i}"): st.session_state.segments.pop(i); st.rerun()

# RENDER
if st.session_state.video_path and st.session_state.segments:
    if st.button("🚀 Render Tutorial Video", type="primary", use_container_width=True):
        with st.status("Assembling tutorial layers..."):
            path = assemble_composition(st.session_state.video_path, st.session_state.segments, voice)
        st.video(path)
        with open(path, "rb") as f: st.download_button("💾 Download Tutorial", f, "tutorial_export.mp4")
