import streamlit as st
import google.generativeai as genai
import os
import time
import asyncio
import edge_tts
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mp

st.set_page_config(page_title="Pro Demo Studio", layout="wide")
st.title("🎬 Professional Instructional Demo Studio")

# --- 1. KEY MANAGEMENT (The "Remember Me" Fix) ---
# This looks for the key in Cloud Secrets or Local Secrets
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    with st.sidebar:
        api_key = st.text_input("Gemini API Key", type="password")
        st.info("To save this key permanently, add it to Streamlit Secrets.")

# --- 2. SETTINGS & MODEL ---
def get_best_model(k):
    try:
        genai.configure(api_key=k)
        models = genai.list_models()
        capable = [m.name for m in models if '1.5' in m.name and 'generateContent' in m.supported_generation_methods]
        return capable[0] if capable else None
    except: return None

async def generate_voice(text, output_path, voice):
    communicate = edge_tts.Communicate(text, voice, rate="+2%")
    await communicate.save(output_path)

def create_text_clip(text, duration, video_size):
    w, h = video_size
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", int(h/22))
    except:
        font = ImageFont.load_default()
    words = text.split()
    lines, cur = [], []
    for word in words:
        cur.append(word)
        if len(" ".join(cur)) > 50:
            lines.append(" ".join(cur[:-1]))
            cur = [word]
    lines.append(" ".join(cur))
    txt = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), txt, font=font, align="center")
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x, y = (w-tw)/2, h*0.82-th
    for o in [(-2,-2), (2,-2), (-2,2), (2,2)]:
        draw.multiline_text((x+o[0], y+o[1]), txt, font=font, fill="black", align="center")
    draw.multiline_text((x, y), txt, font=font, fill="white", align="center")
    return mp.ImageClip(np.array(img)).set_duration(duration)

def assemble_pro_video(v_path, script_data, voice):
    video = mp.VideoFileClip(v_path)
    total_dur = video.duration
    final_clips = []
    last_processed_t = 0
    for i, (s_t, e_t, txt) in enumerate(script_data):
        if s_t > last_processed_t:
            final_clips.append(video.subclip(last_processed_t, s_t))
        s_t = max(0, min(s_t, total_dur - 0.1))
        e_t = min(e_t, total_dur)
        clip = video.subclip(s_t, e_t)
        a_path = f"seg_{i}.mp3"
        asyncio.run(generate_voice(txt, a_path, voice))
        audio = mp.AudioFileClip(a_path)
        if audio.duration > clip.duration:
            last_frame = clip.get_frame(clip.duration - 0.01)
            freeze = mp.ImageClip(last_frame).set_duration(audio.duration - clip.duration)
            seg_vid = mp.concatenate_videoclips([clip, freeze])
        else:
            seg_vid = clip.set_duration(clip.duration)
        sub = create_text_clip(txt, seg_vid.duration, video.size)
        seg_vid = mp.CompositeVideoClip([seg_vid, sub]).set_audio(audio)
        final_clips.append(seg_vid)
        last_processed_t = e_t
    if last_processed_t < total_dur:
        final_clips.append(video.subclip(last_processed_t, total_dur))
    final = mp.concatenate_videoclips(final_clips, method="compose")
    out_file = "final_pro_demo.mp4"
    final.write_videofile(out_file, codec="libx264", audio_codec="aac", fps=24, preset="ultrafast")
    video.close()
    return out_file

# --- 3. UI ---
with st.sidebar:
    voice = st.selectbox("Male Voice", ["en-US-GuyNeural", "en-GB-RyanNeural"])
    if api_key:
        model_name = get_best_model(api_key)
        if model_name:
            st.success(f"Connected: {model_name}")

up = st.file_uploader("Upload Product Video", type=['mp4', 'mov'])

if up and api_key and model_name:
    if st.button("🚀 Generate Full Sync'd Demo"):
        try:
            v_in = "input_video.mp4"
            with open(v_in, "wb") as f:
                f.write(up.read())
            with mp.VideoFileClip(v_in) as temp_clip:
                v_dur_sec = temp_clip.duration
            with st.status("Analyzing...") as status:
                gf = genai.upload_file(path=v_in)
                while gf.state.name == "PROCESSING": time.sleep(2); gf = genai.get_file(gf.name)
                model = genai.GenerativeModel(model_name=model_name)
                prompt = f"""SYSTEM: You are a technical narrator. Cover the full {int(v_dur_sec)}s video. 
                FORMAT: [MM:SS - MM:SS] Text: (Instructional sentence)"""
                resp = model.generate_content([gf, prompt])
                matches = re.findall(r"\[(\d+):(\d+)\s*[-–]\s*(\d+):(\d+)\]\s*Text:\s*(.*)", resp.text)
                data = [(int(m[0])*60+int(m[1]), int(m[2])*60+int(m[3]), m[4]) for m in matches]
                if not data:
                    data = [(0, v_dur_sec, "Walking through the core functionality.")]
                f_out = assemble_pro_video(v_in, data, voice)
                status.update(label="Demo Ready!", state="complete")
            st.video(f_out)
        except Exception as e:
            st.error(f"Error: {e}")
