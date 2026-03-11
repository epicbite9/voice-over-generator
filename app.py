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
from google.api_core import exceptions

# Setup FFmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

# --- CONFIG ---
CONFIG_PATH = ".tutorial_sync_config.json"
OUTPUT_PATH = "premium_tutor_sync.mp4"

# --- UI THEME ---
st.set_page_config(page_title="AI Tutorial Studio Pro", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #212529; }
    .layer-container {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
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
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except: return {}
    return {}

def save_config(data):
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)

def get_valid_model(api_key):
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Prefer flash for speed, then pro
        preferred = [m for m in models if "1.5-flash" in m]
        if not preferred:
            preferred = [m for m in models if "1.5-pro" in m]
        return preferred[0] if preferred else "gemini-1.5-flash"
    except Exception as e:
        return "gemini-1.5-flash"

# --- PREMIUM SYNC ENGINE ---
async def generate_synced_voice(text, path, voice, target_duration):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)
    
    try:
        audio = mp.AudioFileClip(path)
        actual_dur = audio.duration
        audio.close()
        
        safety_target = target_duration * 0.95
        
        if actual_dur > safety_target:
            speed_factor = (actual_dur / safety_target) - 1.0
            rate_percent = int(speed_factor * 100)
            rate_str = f"+{min(rate_percent, 50)}%" 
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            await communicate.save(path)
        elif actual_dur < (target_duration * 0.5):
            communicate = edge_tts.Communicate(text, voice, rate="-10%")
            await communicate.save(path)
    except:
        pass # Fallback to original audio if analysis fails

def parse_ai_response(text):
    try:
        # Find JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match: return []
        
        data = json.loads(match.group())
        raw_segments = data.get("segments", [])
        
        cleaned_segments = []
        for s in raw_segments:
            # SAFETY: Ensure every segment has 'narration' even if AI uses 'text' or 'script'
            narration = s.get("narration") or s.get("text") or s.get("script") or ""
            cleaned_segments.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 5.0)),
                "narration": str(narration)
            })
        return cleaned_segments
    except Exception as e:
        st.error(f"Parsing error: {e}")
        return []

def assemble_composition(video_path, segments, voice):
    render_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join(tempfile.gettempdir(), f"premium_{render_id}")
    os.makedirs(work_dir, exist_ok=True)
    
    video_clip = mp.VideoFileClip(video_path).without_audio()
    v_dur = video_clip.duration
    
    sorted_segs = sorted(segments, key=lambda x: x['start'])
    final_audio_clips = []
    last_audio_end = 0.0

    for i, seg in enumerate(sorted_segs):
        if not seg.get('narration', '').strip(): continue
        
        seg_mp3 = os.path.join(work_dir, f"seg_{i}.mp3")
        target_dur = max(1.0, float(seg['end']) - float(seg['start']))
        
        asyncio.run(generate_synced_voice(seg['narration'], seg_mp3, voice, target_dur))
        
        if os.path.exists(seg_mp3) and os.path.getsize(seg_mp3) > 0:
            a_clip = mp.AudioFileClip(seg_mp3)
            start_time = float(seg['start'])
            
            # Anti-Overlap Logic
            if start_time < last_audio_end:
                start_time = last_audio_end + 0.1
            
            if start_time < v_dur:
                a_clip = a_clip.set_start(start_time)
                final_audio_clips.append(a_clip)
                last_audio_end = start_time + a_clip.duration
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
st.title("🎓 AI Tutorial Studio Pro")

stored = load_config()
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password", value=stored.get("api_key", ""))
    
    # Reset segments if API key changes to prevent mismatch
    if "current_key" not in st.session_state:
        st.session_state.current_key = api_key
    if api_key != st.session_state.current_key:
        st.session_state.segments = []
        st.session_state.current_key = api_key

    voice = st.selectbox("Narrator Voice", ["en-US-AndrewMultilingualNeural", "en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"])
    if st.button("Save Settings"): 
        save_config({"api_key": api_key})
        st.success("Settings saved!")

# LAYER 1: VIDEO
st.markdown('<div class="layer-container video-label"><h3>Layer 1: Tutorial Footage</h3></div>', unsafe_allow_html=True)
up = st.file_uploader("Video", type=["mp4", "mov"], label_visibility="collapsed")
if up:
    if st.session_state.video_path is None or up.name != st.session_state.get('last_fn'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
            t.write(up.getbuffer()); 
            st.session_state.video_path = t.name
            st.session_state['last_fn'] = up.name
        v = mp.VideoFileClip(st.session_state.video_path)
        st.session_state.video_duration = v.duration
        v.close()
    st.video(st.session_state.video_path)

# LAYER 2: AUDIO
st.markdown('<div class="layer-container audio-label"><h3>Layer 2: Sync-Corrected Narration</h3></div>', unsafe_allow_html=True)
if st.button("✨ Generate Synced Tutor Script") and st.session_state.video_path:
    if api_key:
        with st.spinner("AI is analyzing the video..."):
            try:
                model_name = get_valid_model(api_key)
                m = genai.GenerativeModel(model_name)
                vf = genai.upload_file(path=st.session_state.video_path)
                
                # Wait for processing
                while vf.state.name == "PROCESSING":
                    time.sleep(2)
                    vf = genai.get_file(vf.name)
                
                prompt = f"""
                You are a professional TUTOR. Write a narration script for this video.
                Address the visitor as "You". Never use "I".
                Match segments to visual changes.
                Video Duration: {st.session_state.video_duration} seconds.
                
                IMPORTANT: Return ONLY valid JSON in this format:
                {{ "segments": [ {{ "start": 0.0, "end": 4.5, "narration": "Your text here" }} ] }}
                """
                
                res = m.generate_content([vf, prompt])
                parsed = parse_ai_response(res.text)
                if parsed:
                    st.session_state.segments = parsed
                    st.rerun()
                else:
                    st.error("AI returned an invalid format. Please try again.")
                
            except exceptions.ResourceExhausted:
                st.error("🚨 API Limit Reached (Quota full for this key). Please wait 60s or try a different key.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter an API Key in the sidebar first.")

# Display and Edit Segments
if st.session_state.segments:
    # Use a copy to iterate to avoid issues when popping items
    for i in range(len(st.session_state.segments)):
        # Ensure keys exist before rendering
        if 'narration' not in st.session_state.segments[i]:
            st.session_state.segments[i]['narration'] = ""
            
        with st.container():
            st.markdown(f'<div class="segment-box"><b>Segment {i+1}</b>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 1, 4])
            
            # Use .get() and value= for safer data binding
            st.session_state.segments[i]['start'] = c1.number_input(
                "Start(s)", 
                value=float(st.session_state.segments[i].get('start', 0.0)), 
                key=f"s_{i}"
            )
            st.session_state.segments[i]['end'] = c2.number_input(
                "End(s)", 
                value=float(st.session_state.segments[i].get('end', 5.0)), 
                key=f"e_{i}"
            )
            st.session_state.segments[i]['narration'] = c3.text_area(
                "Script", 
                value=st.session_state.segments[i].get('narration', ''), 
                key=f"n_{i}"
            )
            
            if st.button(f"Remove Segment {i+1}", key=f"del_{i}"):
                st.session_state.segments.pop(i)
                st.rerun()

# RENDER
if st.session_state.video_path and st.session_state.segments:
    if st.button("🚀 Render Premium Synced Video", type="primary", use_container_width=True):
        with st.status("Syncing audio and rendering final video..."):
            try:
                path = assemble_composition(st.session_state.video_path, st.session_state.segments, voice)
                st.success("Render complete!")
                st.video(path)
                with open(path, "rb") as f: 
                    st.download_button("💾 Download Synced Tutorial", f, "tutorial_pro.mp4")
            except Exception as e:
                st.error(f"Render failed: {e}")
