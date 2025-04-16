import streamlit as st
import whisper
import tempfile
import os

# Load the model once
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

st.title("🎙️ Whisper Audio Transcriber")
st.write("Upload an audio file and get a text transcription.")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Transcribing..."):
        result = model.transcribe(tmp_path)
        os.remove(tmp_path)  # Clean up temp file

    st.success("Transcription complete!")
    st.text_area("Transcript", result["text"], height=300)

    # Save to file
    filename = uploaded_file.name.rsplit('.', 1)[0] + "_transcript.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(result["text"])

    with open(filename, "r", encoding="utf-8") as f:
        st.download_button("📥 Download Transcript", f, file_name=filename)
