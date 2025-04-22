import os
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset
from morse_alphabet import CHAR_TO_IDX

class MorseDataset(Dataset):
    """
    Dataset for Morse code audio with enriched preprocessing pipeline:
      - Pre-emphasis
      - Silence trimming
      - Bandpass filtering
      - Speed & pitch perturbations
      - Time shift
      - Additive noise & gain
      - Normalization
      - Mel-spectrogram + SpecAugment
    """
    def __init__(self, df, audio_dir, transform=None, sr_target=16000):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.transform = transform or self.default_transform
        self.sr_target = sr_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['id']

        # Load waveform
        path = os.path.join(self.audio_dir, fname)
        waveform, sr = sf.read(path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        # Resample if needed
        if sr != self.sr_target:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sr_target)
            sr = self.sr_target

        # Feature extraction
        features = self.transform(waveform, sr)

        # Encode target
        targets = [CHAR_TO_IDX[c] for c in row['message'] if c in CHAR_TO_IDX]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)

    def default_transform(self, waveform, sr):
        # Pre-emphasis
        pre_emph = 0.97
        waveform = np.append(waveform[0], waveform[1:] - pre_emph * waveform[:-1])

        # Trim silence
        intervals = librosa.effects.split(waveform, top_db=20)
        waveform = np.concatenate([waveform[s:e] for s, e in intervals])

        # Bandpass filter around Morse tone
        low, high = 600, 1200
        b, a = butter(4, [low/(sr/2), high/(sr/2)], btype='band')
        if len(waveform) > max(len(b), len(a)):
            try:
                waveform = filtfilt(b, a, waveform)
            except ValueError:
                pass

        # Speed perturbation
        if np.random.rand() < 0.3:
            rate = np.random.uniform(0.9, 1.1)
            waveform = librosa.effects.time_stretch(waveform, rate)
        # Pitch perturbation
        if np.random.rand() < 0.3:
            n_steps = np.random.uniform(-1, 1)
            waveform = librosa.effects.pitch_shift(waveform, sr, n_steps)

        # Time shift
        if np.random.rand() < 0.5:
            shift = int(np.random.uniform(-0.1, 0.1) * len(waveform))
            waveform = np.roll(waveform, shift)

        # Add noise and random gain
        if np.random.rand() < 0.8:
            waveform += 0.005 * np.random.randn(len(waveform))
        if np.random.rand() < 0.5:
            waveform *= np.random.uniform(0.8, 1.2)

        # Normalize
        waveform = librosa.util.normalize(waveform)

        # Mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=64, n_fft=1024, hop_length=256
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # SpecAugment
        if np.random.rand() < 0.5:
            spec = log_mel.copy()
            # Time mask
            t = np.random.randint(0, spec.shape[1] // 10)
            t0 = np.random.randint(0, spec.shape[1] - t)
            spec[:, t0:t0+t] = spec.mean()
            # Frequency mask
            f = np.random.randint(0, spec.shape[0] // 8)
            f0 = np.random.randint(0, spec.shape[0] - f)
            spec[f0:f0+f, :] = spec.mean()
            log_mel = spec

        return log_mel.T  # (Time, n_mels)


def collate_fn(batch):
    features, targets = zip(*batch)
    input_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    padded_features = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(f) for f in features], batch_first=True
    )
    padded_targets = torch.cat(targets)
    return padded_features, padded_targets, input_lengths, target_lengths
