import asyncio
import os
import json
import re
import tempfile
import time
import edge_tts
import google.generativeai as genai
import imageio_ffmpeg
import moviepy.editor as mp
import numpy as np
import streamlit as st

# Setup FFmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

# --- THEME & CSS ---
st.set_page_config(page_title="AI Video Editor", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* CapCut-style Editor Theme */
    .stApp { background-color: #0e0e10; color: #efeff1; }
    .main { background-color: #0e0e10; }
    
    /* Timeline Container */
    .timeline-track {
        background: #1c1c1f;
        border-radius: 8px;
        padding: 15px;
        margin: 20px 0;
        min-height: 80px;
        display: flex;
        position: relative;
        overflow-x: auto;
        border: 1px solid #333;
    }
    
    /* Individual Clip style */
    .clip-block {
        background: linear-gradient(90deg, #3182ce, #63b3ed);
        border: 1px solid #fff;
        border-radius: 4px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 10px;
        font-weight: bold;
        position: absolute;
        cursor: pointer;
        transition: all 0.2s;
        white-space: nowrap;
        overflow: hidden;
    }
    .clip-block:hover { filter: brightness(1.2); transform: translateY(-2px); }
    .clip-active { border: 2px solid #ffcc00 !important; box-shadow: 0 0 10px #ffcc00; }
    
    /* Control Bar */
    .editor-card {
        background: #18181b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2d2d30;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'segments' not in st.session_state: st.session_state.segments = []
if 'video_path' not in st.session_state: st.session_state.video_path = None
if 'video_duration' not in st.session_state: st.session_state.video_duration = 10.0
if 'selected_idx' not in st.session_state: st.session_state.selected_idx = 0
if 'video_start' not in st.session_state: st.session_state.video_start = 0.0
if 'video_end' not in st.session_state: st.session_state.video_end = 10.0
if 'bg_music' not in st.session_state: st.session_state.bg_music = None

# --- AUDIO ENGINE ---
async def save_voice(text, path, voice, rate, pitch):
    communicate = edge_tts.Communicate(text, voice, rate=f"{int(rate):+d}%", pitch=f"{int(pitch):+d}%")
    await communicate.save(path)

def generate_voice_file(text, voice, target_duration, rate=0, pitch=0):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    asyncio.run(save_voice(text, tmp, voice, rate, pitch))
    return tmp

# --- UI LAYOUT ---
st.title("🎬 AI Editor Pro")

# Top Layout: Preview + Inspector
col_prev, col_insp = st.columns([3, 2])

with col_prev:
    st.markdown("### 📺 Preview")
    up = st.file_uploader("Import Media", type=["mp4", "mov"], label_visibility="collapsed")
    
    if up:
        if st.session_state.video_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
                t.write(up.getbuffer())
                st.session_state.video_path = t.name
            v = mp.VideoFileClip(st.session_state.video_path)
            st.session_state.video_duration = v.duration
            st.session_state.video_end = v.duration
            v.close()
        st.video(st.session_state.video_path)
    else:
        st.info("Please upload a video to begin editing.")

with col_insp:
    st.markdown("### ⚙️ Inspector")
    with st.container(border=True):
        if st.session_state.segments:
            idx = st.session_state.selected_idx
            if idx < len(st.session_state.segments):
                seg = st.session_state.segments[idx]
                st.markdown(f"**Editing Clip #{idx+1}**")
                seg['text'] = st.text_area("Narration Script", seg['text'], key=f"edit_txt_{idx}")
                
                c1, c2 = st.columns(2)
                seg['start'] = c1.number_input("Start Time (s)", 0.0, float(st.session_state.video_duration), float(seg['start']), 0.1, key=f"edit_s_{idx}")
                seg['end'] = c2.number_input("End Time (s)", 0.0, float(st.session_state.video_duration), float(seg['end']), 0.1, key=f"edit_e_{idx}")
                
                v1, v2 = st.columns(2)
                voice_list = ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"]
                seg['voice'] = v1.selectbox("Voice", voice_list, index=voice_list.index(seg['voice']), key=f"edit_v_{idx}")
                seg['rate'] = v2.slider("Speed Offset", -20, 50, int(seg['rate']), key=f"edit_r_{idx}")
                
                if st.button("🗑️ Delete Clip", use_container_width=True):
                    st.session_state.segments.pop(idx)
                    st.rerun()
        else:
            st.write("No clips selected. Add a clip in the timeline.")

# Bottom Layout: Timeline
st.markdown("---")
st.markdown("### 🎞️ Timeline")

# Timeline Visualizer
if st.session_state.video_path:
    total_w = 100 # percentage
    dur = max(1.0, st.session_state.video_duration)
    
    # HTML Timeline Construction
    timeline_html = f'<div class="timeline-track">'
    for i, s in enumerate(st.session_state.segments):
        left = (s['start'] / dur) * 100
        width = ((s['end'] - s['start']) / dur) * 100
        active_class = "clip-active" if i == st.session_state.selected_idx else ""
        timeline_html += f'<div class="clip-block {active_class}" style="left:{left}%; width:{width}%;">Clip {i+1}</div>'
    timeline_html += '</div>'
    st.markdown(timeline_html, unsafe_allow_html=True)

# Timeline Buttons
t_col1, t_col2, t_col3, t_col4 = st.columns([1,1,1,2])
with t_col1:
    if st.button("➕ Add Clip"):
        st.session_state.segments.append({'start': 0.0, 'end': 2.0, 'text': 'Hello world', 'voice': 'en-US-JennyNeural', 'rate': 0, 'pitch': 0})
        st.session_state.selected_idx = len(st.session_state.segments) - 1
        st.rerun()
with t_col2:
    if st.session_state.segments:
        options = [f"Clip {i+1}" for i in range(len(st.session_state.segments))]
        sel = st.selectbox("Select Clip", options, index=st.session_state.selected_idx, label_visibility="collapsed")
        st.session_state.selected_idx = options.index(sel)
with t_col4:
    if st.button("🚀 EXPORT FINAL VIDEO", type="primary", use_container_width=True):
        # ASSEMBLY LOGIC
        with st.status("Assembling Project...") as status:
            video = mp.VideoFileClip(st.session_state.video_path).without_audio()
            voice_clips = []
            temp_files = []
            for i, seg in enumerate(st.session_state.segments):
                if not seg['text'].strip(): continue
                path = generate_voice_file(seg['text'], seg['voice'], 0, seg['rate'], 0)
                temp_files.append(path)
                a_clip = mp.AudioFileClip(path).set_start(seg['start'])
                voice_clips.append(a_clip)
            
            final_audio = mp.CompositeAudioClip(voice_clips).set_duration(video.duration)
            final_video = video.set_audio(final_audio)
            final_video.write_videofile("capcut_export.mp4", codec="libx264", audio_codec="aac", fps=24, logger=None)
            
            st.video("capcut_export.mp4")
            with open("capcut_export.mp4", "rb") as f:
                st.download_button("Download Export", f, "final_video.mp4")
            
            # Cleanup
            video.close()
            for f in temp_files: os.remove(f)

# Footer info
st.sidebar.title("Project Settings")
st.sidebar.markdown("---")
st.sidebar.write(f"Total Clips: {len(st.session_state.segments)}")
if st.sidebar.button("Reset Project"):
    st.session_state.segments = []
    st.session_state.video_path = None
    st.rerun()
