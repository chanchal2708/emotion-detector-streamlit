import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import joblib
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr

# Paths
AUDIO_PATH = "realtime_audio.wav"
MODEL_PATH = "model/audio_emotion_model.pkl"

def record_audio():
    try:
        duration = 4  # seconds
        fs = 44100
        messagebox.showinfo("Recording", "Recording started for 4 seconds...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        write(AUDIO_PATH, fs, recording)
        messagebox.showinfo("Done", "Recording completed!")
    except Exception as e:
        messagebox.showerror("Error", f"Recording failed: {e}")

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.AudioFile(AUDIO_PATH) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except:
        return "Speech not recognized."

def visualize_emotion(emotion):
    emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
    values = [1 if e == emotion else 0 for e in emotions]
    plt.bar(emotions, values, color='coral')
    plt.title(f"Detected Emotion: {emotion}")
    plt.ylim(0, 1.2)
    plt.ylabel("Confidence")
    plt.show()

def predict():
    try:
        features = extract_features(AUDIO_PATH)
        model = joblib.load(MODEL_PATH)
        predicted = model.predict(features)[0]
        text = transcribe_audio()
        
        result_label.config(text=f"Predicted Emotion: {predicted}")
        transcript_label.config(text=f"Transcript: {text}")
        visualize_emotion(predicted)
    except FileNotFoundError:
        messagebox.showerror("Error", "Audio file or model not found.")
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

# GUI Setup
root = tk.Tk()
root.title("Real-Time Speech Emotion Recognition")
root.geometry("500x300")
root.config(bg="white")

tk.Label(root, text="🎤 Speech Emotion Recognizer", font=("Arial", 18, "bold"), bg="white").pack(pady=10)

tk.Button(root, text="Record Audio", command=record_audio, bg="skyblue", width=20).pack(pady=10)
tk.Button(root, text="Predict Emotion", command=predict, bg="lightgreen", width=20).pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14), bg="white", fg="blue")
result_label.pack(pady=10)

transcript_label = tk.Label(root, text="", font=("Arial", 12), bg="white", wraplength=400)
transcript_label.pack(pady=5)

root.mainloop()
