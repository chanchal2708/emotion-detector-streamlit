import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import speech_recognition as sr

def record_audio(filename="realtime_audio.wav", duration=4, fs=44100):
    print("🎤 Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    print("✅ Done Recording")

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

def transcribe_audio(audio_path="realtime_audio.wav"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        print(f"\n📝 Transcribed Text: {text}")
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
    return ""

def visualize_emotion(predicted_emotion):
    emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
    values = [1 if e == predicted_emotion else 0 for e in emotions]
    
    plt.bar(emotions, values, color='skyblue')
    plt.title(f"Detected Emotion: {predicted_emotion}")
    plt.ylabel("Confidence")
    plt.ylim(0, 1.2)
    plt.show()

def predict_emotion():
    record_audio()
    features = extract_features("realtime_audio.wav")
    model = joblib.load("model/audio_emotion_model.pkl")
    prediction = model.predict(features)[0]
    print(f"\n🎯 Predicted Emotion: {prediction}")
    
    transcribe_audio("realtime_audio.wav")
    visualize_emotion(prediction)
    
    return prediction

if __name__ == "__main__":
    predicted_emotion = predict_emotion()
    with open("prediction_result.txt", "w") as f:
        f.write(predicted_emotion)
