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

    def _extract_instrument_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract instrument-specific features for LLM classification."""
        features = {}

        try:
            # Harmonic-to-noise ratio
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_power = np.sum(harmonic ** 2)
            percussive_power = np.sum(percussive ** 2)
            features["harmonic_to_noise_ratio"] = float(
                harmonic_power / (percussive_power + 1e-10)
            )

            # Attack/decay characteristics
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env, sr=sr
            )

            if len(onset_frames) > 0:
                # Calculate attack times (time to peak after onset)
                hop_length = 512
                attack_times = []
                for onset in onset_frames[:10]:  # Analyze first 10 onsets
                    start = onset
                    end = min(onset + 20, len(onset_env))
                    if end > start:
                        segment = onset_env[start:end]
                        peak_pos = np.argmax(segment)
                        attack_time = (peak_pos * hop_length) / sr
                        attack_times.append(attack_time)

                features["avg_attack_time"] = float(np.mean(attack_times)) if attack_times else 0.0
                features["attack_variance"] = float(np.var(attack_times)) if attack_times else 0.0
            else:
                features["avg_attack_time"] = 0.0
                features["attack_variance"] = 0.0

            # Frequency band energy distribution
            stft = librosa.stft(audio)
            power_spectrum = np.abs(stft) ** 2
            freqs = librosa.fft_frequencies(sr=sr)

            # Define frequency bands
            bands = {
                "sub_bass": (20, 60),
                "bass": (60, 250),
                "low_mid": (250, 500),
                "mid": (500, 2000),
                "high_mid": (2000, 4000),
                "presence": (4000, 6000),
                "brilliance": (6000, 20000)
            }

            total_energy = np.sum(power_spectrum)

            for band_name, (low_freq, high_freq) in bands.items():
                # Find frequency bin indices
                low_bin = np.argmax(freqs >= low_freq)
                high_bin = np.argmax(freqs >= high_freq)
                if high_bin == 0:
                    high_bin = len(freqs) - 1

                # Calculate energy in this band
                band_energy = np.sum(power_spectrum[low_bin:high_bin, :])
                features[f"{band_name}_energy"] = float(band_energy / (total_energy + 1e-10))

            # Timbre descriptors
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            for i in range(contrast.shape[0]):
                features[f"spectral_contrast_band_{i}"] = float(np.mean(contrast[i]))

            # Tonnetz (tonal centroid features)
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features["tonnetz_mean"] = float(np.mean(tonnetz))
            features["tonnetz_std"] = float(np.std(tonnetz))

            # Additional timbre features
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
            features["spectral_bandwidth_std"] = float(np.std(bandwidth))

            # Spectral flatness (distinguishes noise vs tonal)
            flatness = librosa.feature.spectral_flatness(y=audio)
            features["spectral_flatness"] = float(np.mean(flatness))

            # Onset density (rhythmic characteristic)
            features["onset_density"] = len(onset_frames) / (len(audio) / sr)

            # Pitch stability (for melodic instruments)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            # Get pitch values where magnitude is significant
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                if magnitudes[index, t] > np.max(magnitudes) * 0.1:
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)

            if pitch_values:
                features["pitch_stability"] = float(1.0 / (np.std(pitch_values) + 1e-10))
                features["dominant_pitch"] = float(np.median(pitch_values))
            else:
                features["pitch_stability"] = 0.0
                features["dominant_pitch"] = 0.0

        except Exception as e:
            logger.warning(f"Failed to extract instrument features: {e}")

        return features