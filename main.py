import librosa
import numpy as np
import torch
import speechbrain as sb
from scipy.spatial.distance import cosine

# Load pre-trained SpeechBrain speaker embedding model
spkrec = sb.pretrained.EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)  # Resample to 16kHz
    return torch.tensor(audio).unsqueeze(0)  # Convert to tensor


def extract_embedding(audio):
    #Extract speaker embedding using SpeechBrain
    with torch.no_grad():
        emb = spkrec.encode_batch(audio)
    return emb.squeeze().numpy()


def compare_voices(baseline_audio, new_audio):
    # Load and preprocess audio
    wav1 = load_audio(baseline_audio)
    wav2 = load_audio(new_audio)

    # Extract embeddings
    emb1 = extract_embedding(wav1)
    emb2 = extract_embedding(wav2)

    # Compute cosine similarity
    similarity_score = 1 - cosine(emb1, emb2)  # Cosine similarity (1 means identical, 0 means different)

    # Normalize to a probability-like scale (0 to 1)
    probability = (similarity_score + 1) / 2  # Convert range from -1 to 1 into 0 to 1

    return probability *  100


if __name__ == "__main__":
    baseline_audio = "voice samples/sample.wav"
    new_audio = "voice samples/sample4.wav"

    similarity = compare_voices(baseline_audio, new_audio)
    print(f"Probability that the voices are from the same person: {similarity:.2f}%")
