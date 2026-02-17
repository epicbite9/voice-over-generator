import streamlit as st
import google.generativeai as genai
import tempfile
import os
import time
import asyncio
import edge_tts
import re
import moviepy.editor as mp
from moviepy.video.VideoClip import ImageClip

st.set_page_config(page_title="Pro Tutorial Sync", layout="wide")
st.title("🎬 Professional AI Tutorial Sync")
st.markdown("Matching narration perfectly to on-screen actions using Freeze-Frame Sync.")

def get_best_model(api_key):
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        capable = [m.name for m in models if '1.5' in m.name and 'generateContent' in m.supported_generation_methods]
        return capable[0] if capable else None
    except: return None

async def generate_segment_audio(text, output_path, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

# --- THE PRO ASSEMBLY ENGINE ---
def assemble_pro_video(original_video_path, script_data, voice_choice):
    video = mp.VideoFileClip(original_video_path)
    segments = []
    current_audio_time = 0
    
    for i, entry in enumerate(script_data):
        start_t, end_t, text = entry
        
        # 1. Extract the specific video slice for this action
        # Ensure we don't go out of bounds
        end_t = min(end_t, video.duration)
        clip = video.subclip(start_t, end_t)
        
        # 2. Generate audio for this specific slice
        audio_seg_path = f"seg_{i}.mp3"
        asyncio.run(generate_segment_audio(text, audio_seg_path, voice_choice))
        audio_seg = mp.AudioFileClip(audio_seg_path)
        
        # 3. PRO SYNC LOGIC: 
        # If audio is longer than video, freeze the last frame of the video
        if audio_seg.duration > clip.duration:
            freeze_duration = audio_seg.duration - clip.duration
            last_frame = clip.get_frame(clip.duration - 0.01)
            freeze_clip = mp.ImageClip(last_frame).set_duration(freeze_duration)
            clip = mp.concatenate_videoclips([clip, freeze_clip])
        else:
            # If video is longer than audio, it plays normally (natural pauses)
            clip = clip.set_duration(clip.duration)
            
        clip = clip.set_audio(audio_seg)
        segments.append(clip)
    
    final_video = mp.concatenate_videoclips(segments, method="compose")
    output_path = "pro_demo.mp4"
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)
    return output_path

# --- UI ---
with st.sidebar:
    key = st.text_input("Gemini API Key", type="password")
    voice = st.selectbox("Narrator Voice", ["en-US-GuyNeural", "en-US-AvaNeural", "en-GB-SoniaNeural"])
    model_name = get_best_model(key) if key else None

uploaded_video = st.file_uploader("Upload Screen Recording", type=['mp4', 'mov'])

if uploaded_video and key and model_name:
    if st.button("Generate Professional Sync'd Demo"):
        try:
            with st.status("Analyzing & Syncing...") as status:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_video.read())
                    v_path = tmp.name

                # A. Analysis with Time-Anchoring
                st.write("Detecting exact timestamps for actions...")
                genai_file = genai.upload_file(path=v_path)
                while genai_file.state.name == "PROCESSING": time.sleep(2); genai_file = genai.get_file(genai_file.name)
                
                model = genai.GenerativeModel(model_name=model_name)
                # We force Gemini to use a specific format we can parse
                prompt = """Analyze this video. Break it into segments based on visual actions.
                For each segment, provide the start time, end time, and a professional narration sentence.
                Format exactly like this:
                [00:00 - 00:04] Text: Welcome to the dashboard.
                [00:04 - 00:10] Text: Click on the media tab to see your files.
                """
                response = model.generate_content([genai_file, prompt])
                
                # B. Parse the Timestamps
                pattern = r"\[(\d+):(\d+)\s*-\s*(\d+):(\d+)\]\s*Text:\s*(.*)"
                matches = re.findall(pattern, response.text)
                
                script_data = []
                for m in matches:
                    start_sec = int(m[0]) * 60 + int(m[1])
                    end_sec = int(m[2]) * 60 + int(m[3])
                    script_data.append((start_sec, end_sec, m[4]))

                if not script_data:
                    st.error("Failed to parse timestamps. Try again.")
                    st.stop()

                # C. Assemble
                st.write(f"Syncing {len(script_data)} segments...")
                final_out = assemble_pro_video(v_path, script_data, voice)
                status.update(label="Complete!", state="complete")

            st.success("Professional Demo Created!")
            st.video(final_out)
            with open(final_out, "rb") as f:
                st.download_button("Download Pro Demo", f, file_name="Professional_Sync.mp4")

        except Exception as e:
            st.error(f"Error: {e}")
