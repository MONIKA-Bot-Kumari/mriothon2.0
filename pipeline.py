import librosa
import noisereduce as nr

import soundfile as sf
import numpy as np
import whisper


def reduce_noise(input_path, output_path):
    print("Step 1: Noise Reduction Processing...")
    y, sr = librosa.load(input_path, sr=16000)
    reduced_audio = nr.reduce_noise(y=y, sr=sr)
    sf.write(output_path, reduced_audio, sr)
    return output_path


def voice_activity_detection(input_path, output_path):
    print("Step 2: Voice Activity Detection Processing...")
    audio, sr = librosa.load(input_path, sr=16000)
    vad = webrtcvad.Vad(3)  # 0-3, 3 = most aggressive

    frame_duration = 30  # ms
    frame_length = int(sr * frame_duration / 1000)

    segments = []

    for i in range(0, len(audio), frame_length):
        frame = audio[i:i + frame_length]
        if len(frame) < frame_length:
            break
        is_speech = vad.is_speech((frame * 32768).astype(np.int16).tobytes(), sr)
        if is_speech:
            segments.extend(frame)

    sf.write(output_path, segments, sr)
    return output_path


def transcribe_audio(audio_path):
    print("Step 3: Speech Recognition (Whisper) Processing...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]


def pipeline(input_clip):
    cleaned = reduce_noise(input_clip, "audio_samples/cleaned.wav")
    voice_only = voice_activity_detection(cleaned, "audio_samples/voice_only.wav")
    text = transcribe_audio(voice_only)

    print("\n================ FINAL OUTPUT ================")
    print("Detected Voice Text: ", text)
    print("================================================")
    return text


if __name__ == "__main__":
    pipeline("audio_samples/test.wav")