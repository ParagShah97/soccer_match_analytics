import streamlit as st
import tempfile
import os
from pathlib import Path
import shutil

# Simulate annotation function
def annotate_video(input_path, output_path):
    shutil.copy(input_path, output_path)

st.set_page_config(page_title="Soccer Match Analyzer", layout="centered")
st.title("⚽ AI Soccer Match Analyzer")

# File uploader
uploaded_file = st.file_uploader("Upload a soccer match video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / uploaded_file.name
        output_path = Path(tmpdir) / f"annotated_{uploaded_file.name}"

        # Save uploaded file
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        # Button to run annotation
        if st.button("Annotate Video"):
            with st.spinner("Processing video..."):
                annotate_video(str(input_path), str(output_path))
            st.success("Annotation complete! Here's the output:")
            st.video(str(output_path))
