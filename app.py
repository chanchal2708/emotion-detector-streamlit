import streamlit as st
import subprocess

st.title("🎤 Real-time Emotion Detector")

if st.button("Predict Emotion"):
    with st.spinner("Recording and analyzing..."):
        result = subprocess.run(
            ["python", "scripts/predict_realtime.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            try:
                with open("prediction_result.txt", "r") as f:
                    prediction = f.read()
                st.success(f"Predicted Emotion: {prediction}")
            except:
                st.error("Could not read prediction result.")
        else:
            st.error("Prediction script failed to run.")
