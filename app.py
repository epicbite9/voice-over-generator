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
    st.session_state.video_duration = 0
if 'bg_music' not in st.session_state:
    st.session_state.bg_music = None
if 'bg_music_volume' not in st.session_state:
    st.session_state.bg_music_volume = 0.15
if 'video_start' not in st.session_state:
    st.session_state.video_start = 0
if 'video_end' not in st.session_state:
    st.session_state.video_end = 0
if 'processed_audio_files' not in st.session_state:
    st.session_state.processed_audio_files = {}

st.set_page_config(page_title="Tutorial Sync Studio", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px;}
.main-content {display: flex; gap: 1rem;}
/* Hero */
.hero {
    background: linear-gradient(135deg, #0a2540 0%, #0f3b61 60%, #155987 100%);
    color: #f8fbff;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}
.hero h1 {margin: 0 0 0.35rem 0; font-size: 1.55rem;}
.hero p {margin: 0; opacity: 0.95;}
/* Cards */
.card {
    background: #fff;
    border: 1px solid #e0e4e8;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.card h3 {margin: 0 0 0.75rem 0; font-size: 1rem; color: #1a202c;}
/* Segment Editor */
.segment-item {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.segment-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
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
.segment-time {
    font-size: 0.8rem;
    color: #718096;
}
.segment-text {
    width: 100%;
    min-height: 60px;
    padding: 0.5rem;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    font-size: 0.9rem;
    resize: vertical;
}
.segment-controls {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
}
.segment-control-group {
    display: flex;
    align-items: center;
    gap: 0.25rem;
}
.segment-control-group label {
    font-size: 0.75rem;
    color: #718096;
}
.segment-control-group input[type="range"] {
    width: 80px;
}
.segment-control-group select {
    font-size: 0.8rem;
    padding: 0.2rem;
}
.segment-control-group input[type="number"] {
    width: 60px;
    font-size: 0.8rem;
    padding: 0.2rem;
}
/* Timeline */
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
    cursor: pointer;
    transition: opacity 0.2s;
}
.timeline-segment:hover {opacity: 0.8;}
/* Buttons */
.btn-primary {
    background: #3182ce;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
}
.btn-primary:hover {background: #2b6cb0;}
.btn-secondary {
    background: #edf2f7;
    color: #2d3748;
    border: 1px solid #e2e8f0;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
}
.btn-secondary:hover {background: #e2e8f0;}
.btn-danger {
    background: #e53e3e;
    color: white;
    border: none;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8rem;
}
.btn-danger:hover {background: #c53030;}
.btn-success {
    background: #38a169;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
}
.btn-success:hover {background: #2f855a;}
/* Sidebar */
.stSidebar {background: #f7fafc;}
/* Tabs */
.stTabs [data-baseweb="tab-list"] {gap: 0.5rem;}
.stTabs [data-baseweb="tab"] {
    padding: 0.5rem 1rem;
    border-radius: 6px 6px 0 0;
}
/* Video preview */
.video-preview {
    background: #000;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 1rem;
}
/* Status messages */
.stSuccess, .stError, .stWarning {border-radius: 8px;}
/* Slider customization */
.stSlider [data-baseweb="slider"] {
    padding-top: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>Tutorial Sync Studio</h1>
  <p>Complete video editing suite with AI-powered narration and full control over your tutorial videos.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ==================== CONFIG ==================== #

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
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
            m.name for m in models
            if "generateContent" in getattr(m, "supported_generation_methods", [])
            and "gemini" in m.name.lower()
        ]
        preferred = [m for m in capable if "1.5" in m or "2.0" in m or "2.5" in m]
        return preferred[0] if preferred else (capable[0] if capable else None)
    except Exception:
        return None


# ==================== AUDIO ==================== #

async def generate_segment_audio_with_rate(text, output_path, voice, rate_percent, pitch_percent=0):
    rate_percent = max(MIN_RATE_PERCENT, min(MAX_RATE_PERCENT, int(rate_percent)))
    pitch_percent = max(-50, min(50, int(pitch_percent)))
    rate = f"{rate_percent:+d}%"
    pitch = f"{pitch_percent:+d}%"
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_path)


def measure_audio_duration(audio_path):
    audio = mp.AudioFileClip(audio_path)
    duration = float(audio.duration)
    audio.close()
    return duration


def build_timed_audio(text, audio_path, voice_choice, target_duration, rate=0, pitch=0):
    rate_percent = rate
    for _ in range(3):
        asyncio.run(
            generate_segment_audio_with_rate(
                text, audio_path, voice_choice, rate_percent, pitch
            )
        )
        current_duration = measure_audio_duration(audio_path)
        diff = abs(current_duration - target_duration)
        if diff <= 0.22:
            break
        speed_factor = current_duration / max(0.2, target_duration)
        desired_adjust = int((speed_factor - 1.0) * 80)
        rate_percent += desired_adjust


def generate_segment_audio(segment_idx, segment_data, voice, temp_dir):
    """Generate and cache audio for a segment"""
    text = segment_data.get('text', '')
    start = segment_data.get('start', 0)
    end = segment_data.get('end', 0)
    rate = segment_data.get('rate', 0)
    pitch = segment_data.get('pitch', 0)
    
    if not text.strip():
        return None
    
    seg_path = os.path.join(temp_dir, f"seg_{segment_idx}.mp3")
    seg_duration = max(0.35, end - start)
    
    try:
        build_timed_audio(text, seg_path, voice, seg_duration, rate, pitch)
        return seg_path
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None


# ==================== SCRIPT PARSING ==================== #

def parse_json_payload(raw_text):
    text = raw_text.strip()
    # Fixed the regex string that was causing the error
    try:
        # First try to find JSON in markdown code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        return json.loads(text)
    except Exception:
        # If that fails, try parsing the whole text
        try:
            return json.loads(text)
        except Exception:
            return {}


def parse_segments_from_response(raw_text, video_duration):
    try:
        payload = parse_json_payload(raw_text)
        segments = payload.get("segments", [])
        result = []
        for s in segments:
            start = float(s.get("start", 0))
            end = float(s.get("end", video_duration * 0.5))
            # Ensure valid timing
            if end <= start:
                end = start + 2.0
            if end > video_duration:
                end = video_duration
            result.append({
                'start': start,
                'end': end,
                'text': s.get("narration", "").strip(),
                'voice': 'en-US-JennyNeural',
                'rate': 0,
                'pitch': 0
            })
        return result
    except Exception:
        return []


# ==================== VIDEO BUILD ==================== #

def assemble_pro_video(original_video_path, script_data, voice_choice, bg_music_path=None, bg_volume=0.15, video_start=0, video_end=None):
    video = mp.VideoFileClip(original_video_path)
    
    # Apply video trimming
    if video_end is None:
        video_end = video.duration
    
    # Trim video to specified range
    if video_start > 0 or video_end < video.duration:
        video = video.subclip(video_start, video_end)
    
    base_video = video.without_audio()
    temp_audio_files = []
    voice_clips = []

    for i, segment in enumerate(script_data):
        text = segment.get('text', '')
        if not text.strip():
            continue
            
        start_t = segment.get('start', 0) - video_start
        end_t = segment.get('end', 0) - video_start
        seg_duration = max(0.35, end_t - start_t)
        
        seg_path = f"seg_{i}.mp3"
        temp_audio_files.append(seg_path)
        
        rate = segment.get('rate', 0)
        pitch = segment.get('pitch', 0)
        
        build_timed_audio(text, seg_path, voice_choice, seg_duration, rate, pitch)

        clip = mp.AudioFileClip(seg_path).set_start(max(0, start_t))
        voice_clips.append(clip)

    # Composite voice track
    if voice_clips:
        voice_track = mp.CompositeAudioClip(voice_clips).set_duration(video.duration)
    else:
        voice_track = None

    # Handle background music
    final_audio = voice_track
    if bg_music_path and os.path.exists(bg_music_path):
        try:
            bg_music = mp.AudioFileClip(bg_music_path)
            # Loop music to match video duration
            if bg_music.duration < video.duration:
                loops = int(np.ceil(video.duration / bg_music.duration))
                bg_music = mp.concatenate_audioclips([bg_music] * loops)
            bg_music = bg_music.subclip(0, video.duration).volumex(bg_volume)
            
            if voice_track:
                # Mix voice and background music
                final_audio = mp.CompositeAudioClip([bg_music, voice_track])
            else:
                final_audio = bg_music
        except Exception as e:
            st.warning(f"Could not add background music: {e}")
            if voice_track:
                final_audio = voice_track

    if final_audio:
        final_video = base_video.set_audio(final_audio)
    else:
        final_video = base_video

    final_video.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac", fps=24, logger=None)

    final_video.close()
    video.close()

    for c in voice_clips:
        try:
            c.close()
        except:
            pass
    for p in temp_audio_files:
        try:
            os.remove(p)
        except:
            pass

    return OUTPUT_PATH


# ==================== SIDEBAR ==================== #

stored = load_config()
default_key = stored.get("gemini_api_key", "")

with st.sidebar:
    st.subheader("Settings")

    key = st.text_input("Gemini API Key", type="password", value=default_key)
    save_key = st.checkbox("Remember API key", value=bool(default_key))

    default_voice = stored.get("default_voice", "en-US-JennyNeural")
    voice_options = [
        "en-US-AndrewMultilingualNeural",
        "en-US-AvaNeural",
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
        "en-AU-NatashaNeural",
        "en-CA-ClaraNeural",
    ]
    try:
        voice_index = voice_options.index(default_voice)
    except:
        voice_index = 3
    
    default_voice = st.selectbox(
        "Default Narrator Voice",
        voice_options,
        index=voice_index
    )

    if key and save_key:
        save_config({"gemini_api_key": key, "default_voice": default_voice})
    elif not save_key:
        save_config({})

    model_name = get_best_model(key) if key else None
    if key and not model_name:
        st.warning("No compatible Gemini model found.")

    st.divider()
    
    st.subheader("Project")
    if st.button("Clear Project", use_container_width=True):
        st.session_state.segments = []
        st.session_state.video_path = None
        st.session_state.video_duration = 0
        st.session_state.bg_music = None
        st.session_state.processed_audio_files = {}
        st.rerun()


# ==================== MAIN UI ==================== #

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Video", "Segments", "Audio", "Export"])

# ==================== TAB 1: VIDEO UPLOAD ==================== #
with tab1:
    st.markdown("### Video Upload and Trimming")
    
    uploaded_video = st.file_uploader(
        "Upload Screen Recording",
        type=["mp4", "mov", "mkv", "avi", "webm"],
        help="Upload your tutorial video file"
    )

    if uploaded_video:
        # Save video to temp file
        if st.session_state.video_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.getbuffer())
                st.session_state.video_path = tmp.name
            
            # Get video duration
            video = mp.VideoFileClip(st.session_state.video_path)
            st.session_state.video_duration = float(video.duration)
            st.session_state.video_end = st.session_state.video_duration
            video.close()
        
        # Display video info
        st.success(f"Video loaded: {st.session_state.video_duration:.1f} seconds")
        
        # Video preview
        st.markdown("#### Video Preview")
        st.video(st.session_state.video_path)
        
        # Video trimming
        st.markdown("#### Video Trimming")
        col1, col2 = st.columns(2)
        with col1:
            video_start = st.slider(
                "Start Time (seconds)",
                0.0, st.session_state.video_duration,
                st.session_state.video_start,
                0.1,
                key="trim_start"
            )
            st.session_state.video_start = video_start
        with col2:
            video_end = st.slider(
                "End Time (seconds)",
                0.0, st.session_state.video_duration,
                st.session_state.video_end,
                0.1,
                key="trim_end"
            )
            st.session_state.video_end = video_end
        
        trimmed_duration = st.session_state.video_end - st.session_state.video_start
        st.info(f"Trimmed video duration: {trimmed_duration:.1f} seconds")
        
        # Clear video button
        if st.button("Remove Video", use_container_width=True):
            st.session_state.video_path = None
            st.session_state.video_duration = 0
            st.session_state.segments = []
            st.session_state.processed_audio_files = {}
            st.rerun()
    else:
        st.info("Upload a video to get started")

# ==================== TAB 2: SEGMENTS ==================== #
with tab2:
    st.markdown("### Narration Segments")
    
    if st.session_state.video_path:
        # Mode selection
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("#### Segment Editor")
        with col2:
            if st.button("Add Segment", use_container_width=True):
                # Add new segment
                last_end = st.session_state.segments[-1]['end'] if st.session_state.segments else 0
                new_segment = {
                    'start': last_end,
                    'end': min(last_end + 2.0, st.session_state.video_duration),
                    'text': '',
                    'voice': default_voice,
                    'rate': 0,
                    'pitch': 0
                }
                st.session_state.segments.append(new_segment)
                st.rerun()
        with col3:
            if st.button("AI Generate", use_container_width=True, type="primary"):
                # Trigger AI generation
                st.session_state.run_ai_generate = True
                st.rerun()
        
        # AI Generation
        if 'run_ai_generate' in st.session_state and st.session_state.run_ai_generate:
            if key and model_name:
                with st.status("Analyzing video and generating narration..."):
                    try:
                        video = mp.VideoFileClip(st.session_state.video_path)
                        duration = float(video.duration)
                        video.close()

                        genai_file = genai.upload_file(path=st.session_state.video_path)
                        while genai_file.state.name == "PROCESSING":
                            time.sleep(2)
                            genai_file = genai.get_file(genai_file.name)

                        model = genai.GenerativeModel(model_name=model_name)

                        prompt = f"""
Analyze this UI tutorial video and return JSON only.

Split into action-based segments.
Each segment must have:
- start (seconds)
- end (seconds)
- narration

Total video duration: {duration:.2f} seconds.

Return format:
{{
  "segments": [
    {{"start": 0.0, "end": 3.8, "narration": "..."}}
  ]
}}
"""

                        response = model.generate_content([genai_file, prompt])
                        new_segments = parse_segments_from_response(response.text, duration)
                        
                        if new_segments:
                            st.session_state.segments = new_segments
                            st.success(f"Generated {len(new_segments)} segments!")
                        else:
                            st.error("Could not generate segments")
                        
                        del st.session_state.run_ai_generate
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        if 'run_ai_generate' in st.session_state:
                            del st.session_state.run_ai_generate
            else:
                st.warning("Please enter your Gemini API key in the sidebar")
                if 'run_ai_generate' in st.session_state:
                    del st.session_state.run_ai_generate
        
        # Timeline visualization
        if st.session_state.segments:
            st.markdown("#### Timeline")
            timeline_html = """
            <div class="timeline-container">
                <div class="timeline-bar">
            """
            
            total_duration = st.session_state.video_end - st.session_state.video_start
            for i, seg in enumerate(st.session_state.segments):
                start_pct = ((seg['start'] - st.session_state.video_start) / total_duration) * 100
                width_pct = ((seg['end'] - seg['start']) / total_duration) * 100
                start_pct = max(0, min(100, start_pct))
                width_pct = max(2, min(100 - start_pct, width_pct))
                
                timeline_html += f"""
                    <div class="timeline-segment" style="left: {start_pct}%; width: {width_pct}%;" title="Segment {i+1}: {seg['text'][:30]}...">
                        {i+1}
                    </div>
                """
            
            timeline_html += """
                </div>
            </div>
            """
            st.markdown(timeline_html, unsafe_allow_html=True)
            
            # Segment list
            st.markdown("#### Segments")
            
            # Container for editing
            segments_to_remove = []
            
            for i, segment in enumerate(st.session_state.segments):
                with st.container():
                    st.markdown(f"""
                    <div class="segment-item">
                        <div class="segment-header">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <span class="segment-number">{i+1}</span>
                                <span class="segment-time">{segment['start']:.1f}s to {segment['end']:.1f}s</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Text editing
                    new_text = st.text_area(
                        "Narration Text",
                        value=segment.get('text', ''),
                        height=80,
                        key=f"text_{i}",
                        placeholder="Enter narration text for this segment..."
                    )
                    st.session_state.segments[i]['text'] = new_text
                    
                    # Timing controls
                    col_a, col_b = st.columns(2)
                    with col_a:
                        new_start = st.slider(
                            "Start (s)",
                            0.0, st.session_state.video_duration,
                            segment.get('start', 0),
                            0.1,
                            key=f"start_{i}"
                        )
                        st.session_state.segments[i]['start'] = new_start
                    with col_b:
                        new_end = st.slider(
                            "End (s)",
                            0.0, st.session_state.video_duration,
                            segment.get('end', 0),
                            0.1,
                            key=f"end_{i}"
                        )
                        st.session_state.segments[i]['end'] = new_end
                    
                    # Voice settings
                    st.markdown("##### Voice Settings")
                    col_v1, col_v2, col_v3 = st.columns(3)
                    with col_v1:
                        new_voice = st.selectbox(
                            "Voice",
                            voice_options,
                            index=3,
                            key=f"voice_{i}"
                        )
                        st.session_state.segments[i]['voice'] = new_voice
                    with col_v2:
                        new_rate = st.slider(
                            "Speed (%)",
                            -15, 45,
                            segment.get('rate', 0),
                            1,
                            help="Adjust speaking speed",
                            key=f"rate_{i}"
                        )
                        st.session_state.segments[i]['rate'] = new_rate
                    with col_v3:
                        new_pitch = st.slider(
                            "Pitch (%)",
                            -50, 50,
                            segment.get('pitch', 0),
                            1,
                            help="Adjust voice pitch",
                            key=f"pitch_{i}"
                        )
                        st.session_state.segments[i]['pitch'] = new_pitch
                    
                    # Segment controls
                    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
                    with col_btn1:
                        if st.button(f"Preview #{i+1}", key=f"preview_{i}", use_container_width=True):
                            st.session_state.preview_segment = i
                            st.rerun()
                    with col_btn2:
                        if st.button(f"Regenerate #{i+1}", key=f"regen_{i}", use_container_width=True):
                            st.session_state.regen_segment = i
                            st.rerun()
                    with col_btn3:
                        if st.button(f"Delete #{i+1}", key=f"delete_{i}", use_container_width=True):
                            segments_to_remove.append(i)
                    
                    st.divider()
            
            # Handle segment deletion
            if segments_to_remove:
                for idx in sorted(segments_to_remove, reverse=True):
                    st.session_state.segments.pop(idx)
                st.rerun()
            
            # Preview single segment
            if 'preview_segment' in st.session_state:
                idx = st.session_state.preview_segment
                if idx < len(st.session_state.segments):
                    segment = st.session_state.segments[idx]
                    if segment.get('text', '').strip():
                        with st.spinner("Generating preview audio..."):
                            try:
                                temp_dir = tempfile.mkdtemp()
                                audio_path = generate_segment_audio(idx, segment, segment.get('voice', default_voice), temp_dir)
                                if audio_path and os.path.exists(audio_path):
                                    st.audio(audio_path)
                                    # Clean up after a while
                                    try:
                                        os.remove(audio_path)
                                    except:
                                        pass
                                else:
                                    st.error("Could not generate audio")
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.warning("Please add text to preview")
                del st.session_state.preview_segment
            
            # Regenerate single segment
            if 'regen_segment' in st.session_state:
                idx = st.session_state.regen_segment
                if idx < len(st.session_state.segments):
                    st.session_state.segments[idx]['text'] = f"[Regenerated segment {idx+1}]"
                del st.session_state.regen_segment
                st.rerun()
        
        else:
            st.info("No segments yet. Click 'Add Segment' or use 'AI Generate' to create narration segments.")
    
    else:
        st.info("Please upload a video first in the Video tab")

# ==================== TAB 3: AUDIO ==================== #
with tab3:
    st.markdown("### Audio Settings")
    
    # Background music
    st.markdown("#### Background Music")
    
    bg_music_upload = st.file_uploader(
        "Upload Background Music",
        type=["mp3", "wav", "ogg", "m4a"],
        help="Add background music to your video"
    )
    
    if bg_music_upload:
        if st.session_state.bg_music is None or st.session_state.get('bg_music_changed'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(bg_music_upload.getbuffer())
                st.session_state.bg_music = tmp.name
                st.session_state.bg_music_changed = False
    
    if st.session_state.bg_music and os.path.exists(st.session_state.bg_music):
        col_bm1, col_bm2 = st.columns([3, 1])
        with col_bm1:
            st.success("Background music loaded")
            st.audio(st.session_state.bg_music)
        with col_bm2:
            if st.button("Remove Music", use_container_width=True):
                st.session_state.bg_music = None
                st.rerun()
        
        bg_volume = st.slider(
            "Background Music Volume",
            0.0, 1.0,
            st.session_state.bg_music_volume,
            0.05,
            key="bg_volume"
        )
        st.session_state.bg_music_volume = bg_volume
    else:
        st.info("No background music uploaded")
    
    st.divider()
    
    # Master voice settings
    st.markdown("#### Master Voice Settings")
    col_mv1, col_mv2 = st.columns(2)
    with col_mv1:
        master_voice = st.selectbox(
            "Apply to All Segments - Voice",
            voice_options,
            index=3,
            key="master_voice"
        )
    with col_mv2:
        master_rate = st.slider(
            "Apply to All Segments - Speed (%)",
            -15, 45,
            0,
            1,
            key="master_rate"
        )
    
    if st.button("Apply to All Segments", use_container_width=True):
        for seg in st.session_state.segments:
            seg['voice'] = master_voice
            seg['rate'] = master_rate
        st.success("Applied voice settings to all segments")
        st.rerun()

# ==================== TAB 4: EXPORT ==================== #
with tab4:
    st.markdown("### Export Video")
    
    if st.session_state.video_path and st.session_state.segments:
        # Show preview of segments
        st.markdown("#### Segment Summary")
        total_text = sum(len(s.get('text', '')) for s in st.session_state.segments)
        total_duration = sum(s.get('end', 0) - s.get('start', 0) for s in st.session_state.segments)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Segments", len(st.session_state.segments))
        with col_s2:
            st.metric("Total Narration", f"{total_text} chars")
        with col_s3:
            st.metric("Total Duration", f"{total_duration:.1f}s")
        
        # Check for empty segments
        empty_segments = [i for i, s in enumerate(st.session_state.segments) if not s.get('text', '').strip()]
        if empty_segments:
            st.warning(f"Segments {', '.join(str(i+1) for i in empty_segments)} have no text!")
        
        st.markdown("---")
        
        # Export settings
        st.markdown("#### Export Settings")
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            output_voice = st.selectbox(
                "Voice for Export",
                voice_options,
                index=3,
                key="export_voice"
            )
        with col_e2:
            output_fps = st.selectbox(
                "Frame Rate",
                [24, 30, 60],
                index=0,
                key="export_fps"
            )
        
        # Export button
        st.markdown("---")
        
        if st.button("Generate Final Video", type="primary", use_container_width=True):
            with st.status("Generating video...") as status:
                try:
                    # Update voice for all segments to selected export voice
                    export_segments = []
                    for seg in st.session_state.segments:
                        export_seg = seg.copy()
                        export_seg['voice'] = output_voice
                        export_segments.append(export_seg)
                    
                    final_out = assemble_pro_video(
                        st.session_state.video_path,
                        export_segments,
                        output_voice,
                        st.session_state.bg_music,
                        st.session_state.bg_music_volume,
                        st.session_state.video_start,
                        st.session_state.video_end
                    )
                    
                    status.update(label="Complete", state="complete")
                    
                    st.success("Video generated successfully!")
                    st.video(final_out)
                    
                    with open(final_out, "rb") as f:
                        st.download_button(
                            "Download Final Video",
                            f,
                            file_name="tutorial_sync_final.mp4",
                            use_container_width=True,
                        )
                        
                except Exception as e:
                    st.error(f"Error generating video: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    elif not st.session_state.video_path:
        st.info("Please upload a video first")
    elif not st.session_state.segments:
        st.info("Please create segments first in the Segments tab")
