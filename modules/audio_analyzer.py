import torch
import torchaudio
import numpy as np
from typing import Dict, Optional
import logging

# Import librosa for advanced audio analysis

logger = logging.getLogger(__name__)
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("Librosa not installed. Install with: pip install librosa")
    LIBROSA_AVAILABLE = False


class AudioAnalyzer:
    """Analyze audio features for educational insights."""

    def __init__(self, config: Dict):
        self.config = config
        self.sample_rate = config.get("sample_rate", 44100)

    def analyze(self, waveform: torch.Tensor, sample_rate: int) -> Dict:
        """
        Perform comprehensive audio analysis.

        Returns:
            Dictionary of audio features and metrics
        """
        features = {}

        # Convert to numpy for librosa
        if waveform.dim() > 1:
            audio_np = waveform.mean(dim=0).numpy()  # Convert to mono
        else:
            audio_np = waveform.numpy()

        # Basic features
        features["duration"] = len(audio_np) / sample_rate
        features["rms_energy"] = float(torch.sqrt(torch.mean(waveform ** 2)))
        features["peak_amplitude"] = float(torch.max(torch.abs(waveform)))

        if LIBROSA_AVAILABLE:
            # Spectral features
            features.update(self._extract_spectral_features(audio_np, sample_rate))

            # Rhythm features
            features.update(self._extract_rhythm_features(audio_np, sample_rate))

            # Harmonic features
            features.update(self._extract_harmonic_features(audio_np, sample_rate))

        return features

    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract spectral features using librosa."""
        features = {}

        try:
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features["spectral_centroid_mean"] = float(np.mean(centroid))
            features["spectral_centroid_std"] = float(np.std(centroid))

            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features["spectral_rolloff_mean"] = float(np.mean(rolloff))

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features["zero_crossing_rate"] = float(np.mean(zcr))

            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))

        except Exception as e:
            logger.warning(f"Failed to extract spectral features: {e}")

        return features

    def _extract_rhythm_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract rhythm and tempo features."""
        features = {}

        try:
            # Tempo estimation
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features["tempo_bpm"] = float(tempo)
            features["num_beats"] = len(beats)

            # Beat strength
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            features["onset_strength_mean"] = float(np.mean(onset_env))

        except Exception as e:
            logger.warning(f"Failed to extract rhythm features: {e}")

        return features

    def _extract_harmonic_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract harmonic and pitch features."""
        features = {}

        try:
            # Harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            features["harmonic_ratio"] = float(
                np.sum(harmonic ** 2) / (np.sum(audio ** 2) + 1e-10)
            )

            # Pitch estimation (simplified)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_mean = np.mean(pitches[magnitudes > np.max(magnitudes) * 0.1])
            if not np.isnan(pitch_mean):
                features["estimated_pitch_hz"] = float(pitch_mean)

        except Exception as e:
            logger.warning(f"Failed to extract harmonic features: {e}")

        return features
