import streamlit as st
import tempfile
from pathlib import Path
import shutil

# Simulated annotation function
def process_video(input_path, output_path):
    shutil.copy(input_path, output_path)

st.set_page_config(page_title="Soccer Match Analyzer", layout="centered")
st.title("⚽ AI Soccer Match Analyzer")

# Path to default sample video
DEFAULT_VIDEO = "test_video.mp4" 

# Initialize session state for video path
if "video_path" not in st.session_state:
    st.session_state.video_path = DEFAULT_VIDEO

# Show the current video (default or uploaded/processed)
st.video(st.session_state.video_path)

# File uploader
uploaded_file = st.file_uploader("Upload a soccer match video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        st.session_state.video_path = tmp_file.name
        st.success("Video uploaded successfully!")
        st.video(st.session_state.video_path)

# Button to run analysis
if st.button("AI Analyze"):
    with st.spinner("Processing video..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_file:
            process_video(st.session_state.video_path, output_file.name)
            st.session_state.video_path = output_file.name
    st.success("Annotation complete! Here's the output:")
    st.video(st.session_state.video_path)
