import asyncio
import json
import os
import re
import tempfile
import time
import edge_tts
import google.generativeai as genai
import imageio_ffmpeg
import moviepy.editor as mp
import numpy as np
import streamlit as st

# Setup FFmpeg path
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

CONFIG_PATH = ".tutorial_sync_config.json"
OUTPUT_PATH = "final_output.mp4"

# Initialize session state
if 'segments' not in st.session_state: st.session_state.segments = []
if 'video_path' not in st.session_state: st.session_state.video_path = None
if 'video_duration' not in st.session_state: st.session_state.video_duration = 0.0
if 'bg_music' not in st.session_state: st.session_state.bg_music = None
if 'bg_music_volume' not in st.session_state: st.session_state.bg_music_volume = 0.15
if 'video_start' not in st.session_state: st.session_state.video_start = 0.0
if 'video_end' not in st.session_state: st.session_state.video_end = 0.0

st.set_page_config(page_title="AI Tutorial Sync", layout="wide")

# ==================== VOICE OVER ENGINE ==================== #

async def save_voice(text, path, voice, rate, pitch):
    """Core function to generate the MP3 file via edge-tts"""
    communicate = edge_tts.Communicate(text, voice, rate=f"{int(rate):+d}%", pitch=f"{int(pitch):+d}%")
    await communicate.save(path)

def generate_voice_segment(text, voice, target_duration, rate_offset=0, pitch=0):
    """Generates audio and stretches/shrinks it to fit the segment timing"""
    temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    
    # Attempt to match duration by adjusting rate
    current_rate = rate_offset
    for attempt in range(2):
        asyncio.run(save_voice(text, temp_mp3, voice, current_rate, pitch))
        audio_clip = mp.AudioFileClip(temp_mp3)
        actual_dur = audio_clip.duration
        audio_clip.close()
        
        if abs(actual_dur - target_duration) < 0.3:
            break
        # Calculate new rate to match target
        speed_factor = actual_dur / max(0.1, target_duration)
        current_rate += int((speed_factor - 1.0) * 70)
    
    return temp_mp3

# ==================== VIDEO ASSEMBLY ==================== #

def assemble_video():
    if not st.session_state.video_path or not st.session_state.segments:
        return None

    with st.status("🎬 Processing Video & Generating Voice-overs...") as status:
        # 1. Load and trim base video
        video = mp.VideoFileClip(st.session_state.video_path)
        video = video.subclip(st.session_state.video_start, st.session_state.video_end).without_audio()
        
        voice_clips = []
        temp_files = []

        # 2. Generate and place audio for each segment
        for i, seg in enumerate(st.session_state.segments):
            if not seg['text'].strip(): continue
            
            status.write(f"Generating audio for Segment {i+1}...")
            dur = max(0.2, seg['end'] - seg['start'])
            
            # Create the audio file
            mp3_path = generate_voice_segment(seg['text'], seg['voice'], dur, seg['rate'], seg['pitch'])
            temp_files.append(mp3_path)
            
            # Create moviepy clip
            a_clip = mp.AudioFileClip(mp3_path)
            # Position relative to trimmed video start
            start_rel = max(0, seg['start'] - st.session_state.video_start)
            a_clip = a_clip.set_start(start_rel)
            voice_clips.append(a_clip)

        # 3. Combine Audio
        if voice_clips:
            voice_track = mp.CompositeAudioClip(voice_clips).set_duration(video.duration)
        else:
            voice_track = None

        # 4. Background Music
        final_audio = voice_track
        if st.session_state.bg_music and os.path.exists(st.session_state.bg_music):
            bg = mp.AudioFileClip(st.session_state.bg_music)
            if bg.duration < video.duration:
                bg = mp.concatenate_audioclips([bg] * int(np.ceil(video.duration / bg.duration)))
            bg = bg.subclip(0, video.duration).volumex(st.session_state.bg_music_volume)
            final_audio = mp.CompositeAudioClip([bg, voice_track]) if voice_track else bg

        # 5. Write Output
        final_video = video.set_audio(final_audio)
        final_video.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac", fps=24, logger=None)
        
        # Cleanup
        video.close()
        final_video.close()
        for f in temp_files: 
            try: os.remove(f)
            except: pass
            
        return OUTPUT_PATH

# ==================== STREAMLIT UI ==================== #

st.title("🎙️ AI Video Narrator")

with st.sidebar:
    key = st.text_input("Gemini API Key", type="password")
    voice_list = ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-AU-NatashaNeural"]
    sel_voice = st.selectbox("Global Voice", voice_list, index=0)

tabs = st.tabs(["📁 Video", "📝 Segments", "🎶 Audio", "🚀 Export"])

with tabs[0]:
    up = st.file_uploader("Upload Tutorial Video", type=["mp4", "mov"])
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
        
        m_dur = float(st.session_state.video_duration)
        st.session_state.video_start = st.slider("Trim Start", 0.0, m_dur, st.session_state.video_start)
        st.session_state.video_end = st.slider("Trim End", 0.0, m_dur, st.session_state.video_end)

with tabs[1]:
    if st.session_state.video_path:
        if st.button("➕ Add Segment"):
            st.session_state.segments.append({'start': 0.0, 'end': 2.0, 'text': '', 'voice': sel_voice, 'rate': 0, 'pitch': 0})
        
        for i, s in enumerate(st.session_state.segments):
            with st.expander(f"Segment {i+1}"):
                s['text'] = st.text_area("Narration", s['text'], key=f"txt_{i}")
                c1, c2 = st.columns(2)
                s['start'] = c1.number_input("Start (s)", value=float(s['start']), key=f"s_{i}")
                s['end'] = c2.number_input("End (s)", value=float(s['end']), key=f"e_{i}")
                s['voice'] = st.selectbox("Voice", voice_list, index=voice_list.index(s['voice']), key=f"v_{i}")
                if st.button("🗑️ Delete", key=f"del_{i}"):
                    st.session_state.segments.pop(i); st.rerun()

with tabs[2]:
    bg_up = st.file_uploader("Upload Background MP3", type=["mp3"])
    if bg_up:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as t:
            t.write(bg_up.getbuffer())
            st.session_state.bg_music = t.name
    st.session_state.bg_music_volume = st.slider("Music Volume", 0.0, 1.0, 0.15)

with tabs[3]:
    if st.button("🚀 Generate Final Video", type="primary"):
        path = assemble_video()
        if path:
            st.success("Done!")
            st.video(path)
            with open(path, "rb") as f:
                st.download_button("Download Video", f, "tutorial.mp4")
