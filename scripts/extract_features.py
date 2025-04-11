import os
import pandas as pd
import librosa
import numpy as np

# Emotion label mapping from RAVDESS filename
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Function to extract MFCC features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Main function to parse audio and save features
def parse_audio_files(audio_dir):
    features = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")

                # Extract emotion label from filename
                parts = file.split('-')
                if len(parts) > 2:
                    emotion_label = emotion_map.get(parts[2], None)
                    if emotion_label:
                        mfcc = extract_features(file_path)
                        if mfcc is not None:
                            feature_row = [file_path] + list(mfcc) + [emotion_label]
                            features.append(feature_row)

    # Column names: file_path, mfcc1...mfcc40, label
    columns = ['file_path'] + [f'mfcc{i}' for i in range(1, 41)] + ['label']
    df = pd.DataFrame(features, columns=columns)

    # Save features to CSV
    output_path = 'data/processed_features.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Features extracted and saved to {output_path}")

# Run the function
if __name__ == "__main__":
    parse_audio_files("data/audio/")
