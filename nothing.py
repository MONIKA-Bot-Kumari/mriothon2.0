import librosa
import numpy as np
import os
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# -----------------------------
# Step 1: Feature Extraction
# -----------------------------
def extract_mfcc(y, sr, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# -----------------------------
# Step 2: Voice Activity Detection (Energy-based)
# -----------------------------
def detect_voice_segments(y, sr, frame_duration=0.03, energy_threshold=0.01):
    frame_length = int(frame_duration * sr)
    frames = [y[i:i+frame_length] for i in range(0, len(y), frame_length)]
    segments = []
    for i, frame in enumerate(frames):
        energy = np.sum(frame**2)/len(frame)
        if energy > energy_threshold:
            start = i*frame_duration
            end = (i+1)*frame_duration
            segments.append((start, end))
    return segments

# -----------------------------
# Step 3: Noise Classification Model
# -----------------------------
def prepare_noise_dataset(dataset_path="noise_dataset/"):
    features = []
    labels = []
    for noise_type in os.listdir(dataset_path):
        noise_folder = os.path.join(dataset_path, noise_type)
        for file in os.listdir(noise_folder):
            file_path = os.path.join(noise_folder, file)
            y, sr = librosa.load(file_path, sr=16000)
            mfcc = extract_mfcc(y, sr)
            features.append(mfcc)
            labels.append(noise_type)
    X = np.array(features)
    y_encoded = LabelEncoder().fit_transform(labels)
    return X, y_encoded

def build_noise_model(input_shape, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Step 4: Run Pipeline on Clip
# -----------------------------
def process_clip(clip_path):
    y, sr = librosa.load(clip_path, sr=16000)

    # Voice segments
    voice_segments = detect_voice_segments(y, sr)

    # Non-voice segments for noise
    segment_duration = 0.5  # 0.5s segments for noise classification
    noise_segments = []
    for i in range(0, len(y), int(segment_duration*sr)):
        seg = y[i:i+int(segment_duration*sr)]
        start = i/sr
        end = (i+int(segment_duration*sr))/sr
        # skip if overlaps with voice
        if not any(v_start < end and v_end > start for v_start, v_end in voice_segments):
            noise_segments.append((seg, start, end))

    # Load trained noise model
    X_train, y_train = prepare_noise_dataset()
    model = build_noise_model(X_train.shape[1], len(np.unique(y_train)))
    model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

    # Predict noise type
    results = []
    for seg, start, end in noise_segments:
        mfcc = extract_mfcc(seg, sr)[np.newaxis, :]
        pred = model.predict(mfcc)
        label = np.argmax(pred)
        results.append({"start": round(start,2), "end": round(end,2), "noise_type": str(label)})

    # Combine voice segments
    for v_start, v_end in voice_segments:
        results.append({"start": round(v_start,2), "end": round(v_end,2), "voice": True})

    # Save output
    out_path = os.path.join("outputs", os.path.basename(clip_path).replace(".wav", ".json"))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Processed {clip_path}, results saved to {out_path}")

# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    for file in os.listdir("../data"):
        if file.endswith(".wav"):
            process_clip(os.path.join("../data", file))