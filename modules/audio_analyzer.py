import torch
import torchaudio
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("Librosa not installed. Install with: pip install librosa")
    LIBROSA_AVAILABLE = False


class AudioAnalyzer:
    """Enhanced audio analyzer focused on separation quality metrics."""

    def __init__(self, config: Dict):
        self.config = config
        self.sample_rate = config.get("sample_rate", 44100)

    def analyze(self, waveform: torch.Tensor, sample_rate: int) -> Dict:
        """
        Perform comprehensive audio analysis with focus on separation quality.

        Returns:
            Dictionary of audio features and quality metrics
        """
        features = {}

        # Convert to numpy for librosa
        if waveform.dim() > 1:
            audio_np = waveform.mean(dim=0).numpy()  # Convert to mono
        else:
            audio_np = waveform.numpy()

        # Basic audio properties
        features["duration"] = len(audio_np) / sample_rate
        features["sample_rate"] = sample_rate
        features["channels"] = waveform.shape[0] if waveform.dim() > 1 else 1

        # Energy and dynamics
        features["rms_energy"] = float(torch.sqrt(torch.mean(waveform ** 2)))
        features["peak_amplitude"] = float(torch.max(torch.abs(waveform)))
        features["dynamic_range_db"] = 20 * np.log10(features["peak_amplitude"] / (features["rms_energy"] + 1e-10))

        # Signal quality indicators
        features.update(self._extract_quality_features(waveform, audio_np, sample_rate))

        if LIBROSA_AVAILABLE:
            # Advanced spectral features
            features.update(self._extract_spectral_features(audio_np, sample_rate))

            # Rhythm and tempo features
            features.update(self._extract_rhythm_features(audio_np, sample_rate))

            # Harmonic content analysis
            features.update(self._extract_harmonic_features(audio_np, sample_rate))

            # Separation readiness assessment
            features.update(self._assess_separation_readiness(audio_np, sample_rate))

        return features

    def _extract_quality_features(self, waveform: torch.Tensor, audio_np: np.ndarray, sr: int) -> Dict:
        """Extract features relevant to separation quality."""
        features = {}

        # Zero crossing rate (indicates noise/speech content)
        if len(audio_np) > 0:
            zero_crossings = np.where(np.diff(np.signbit(audio_np)))[0]
            features["zero_crossing_rate"] = len(zero_crossings) / len(audio_np)

        # Spectral rolloff and centroid
        fft = torch.fft.rfft(waveform.flatten())
        magnitude = torch.abs(fft)
        freqs = torch.fft.rfftfreq(len(waveform.flatten()), 1 / sr)

        # Spectral centroid (brightness indicator)
        spectral_centroid = torch.sum(freqs * magnitude) / (torch.sum(magnitude) + 1e-10)
        features["spectral_centroid_hz"] = float(spectral_centroid)

        # Spectral rolloff (85% of energy)
        cumulative_energy = torch.cumsum(magnitude ** 2, dim=0)
        total_energy = cumulative_energy[-1]
        rolloff_idx = torch.where(cumulative_energy >= 0.85 * total_energy)[0]
        if len(rolloff_idx) > 0:
            features["spectral_rolloff_hz"] = float(freqs[rolloff_idx[0]])

        # Frequency distribution analysis
        features.update(self._analyze_frequency_distribution(magnitude, freqs))

        return features

    def _analyze_frequency_distribution(self, magnitude: torch.Tensor, freqs: torch.Tensor) -> Dict:
        """Analyze frequency distribution for separation quality assessment."""
        features = {}

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

        total_energy = torch.sum(magnitude ** 2)

        for band_name, (low_freq, high_freq) in bands.items():
            # Find frequency bins within band
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_energy = torch.sum(magnitude[band_mask] ** 2)
            band_ratio = band_energy / (total_energy + 1e-10)
            features[f"{band_name}_energy_ratio"] = float(band_ratio)

        return features

    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract advanced spectral features using librosa."""
        features = {}

        try:
            # Mel-frequency cepstral coefficients
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
                features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

            # Spectral features for separation quality
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
            features["spectral_centroid_std"] = float(np.std(spectral_centroids))

            # Spectral bandwidth (indicates complexity)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))

            # Spectral contrast (indicates separation potential)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features["spectral_contrast_mean"] = float(np.mean(spectral_contrast))
            features["spectral_contrast_std"] = float(np.std(spectral_contrast))

            # Zero crossing rate variability
            zcr = librosa.feature.zero_crossing_rate(audio)
            features["zcr_mean"] = float(np.mean(zcr))
            features["zcr_std"] = float(np.std(zcr))

        except Exception as e:
            logger.warning(f"Failed to extract spectral features: {e}")

        return features

    def _extract_rhythm_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract rhythm and tempo features."""
        features = {}

        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features["tempo_bpm"] = float(tempo)
            features["num_beats"] = len(beats)

            if len(beats) > 1:
                beat_intervals = np.diff(beats) * (sr / 1000)  # Convert to ms
                features["beat_consistency"] = 1.0 / (np.std(beat_intervals) + 1e-10)

            # Onset detection strength
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            features["onset_strength_mean"] = float(np.mean(onset_env))
            features["onset_strength_std"] = float(np.std(onset_env))

            # Rhythmic complexity
            features["rhythmic_complexity"] = float(np.std(onset_env) / (np.mean(onset_env) + 1e-10))

        except Exception as e:
            logger.warning(f"Failed to extract rhythm features: {e}")

        return features

    def _extract_harmonic_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract harmonic content features."""
        features = {}

        try:
            # Harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)

            harmonic_energy = np.sum(harmonic ** 2)
            percussive_energy = np.sum(percussive ** 2)
            total_energy = np.sum(audio ** 2)

            features["harmonic_ratio"] = float(harmonic_energy / (total_energy + 1e-10))
            features["percussive_ratio"] = float(percussive_energy / (total_energy + 1e-10))
            features["harmonic_percussive_ratio"] = float(harmonic_energy / (percussive_energy + 1e-10))

            # Pitch estimation
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)

            # Extract dominant pitches
            pitch_track = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_track.append(pitch)

            if pitch_track:
                features["pitch_mean_hz"] = float(np.mean(pitch_track))
                features["pitch_std_hz"] = float(np.std(pitch_track))
                features["pitch_range_hz"] = float(np.max(pitch_track) - np.min(pitch_track))

            # Tonal complexity
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features["tonal_complexity"] = float(np.std(chroma))

        except Exception as e:
            logger.warning(f"Failed to extract harmonic features: {e}")

        return features

    def _assess_separation_readiness(self, audio: np.ndarray, sr: int) -> Dict:
        """Assess how suitable the audio is for source separation."""
        assessment = {}

        try:
            # Multi-instrument likelihood based on spectral complexity
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            complexity_score = np.mean(np.std(spectral_contrast, axis=1))
            assessment["multi_instrument_likelihood"] = min(complexity_score / 5.0, 1.0)

            # Stereo separation potential (if stereo)
            if len(audio.shape) > 1:
                left_right_correlation = np.corrcoef(audio[0], audio[1])[0, 1]
                assessment["stereo_separation_potential"] = 1.0 - abs(left_right_correlation)

            # Dynamic range assessment
            dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.sqrt(np.mean(audio ** 2)) + 1e-10))
            assessment["dynamic_range_quality"] = min(dynamic_range / 20.0, 1.0)

            # Frequency content richness
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio), 1 / sr)

            # Count significant frequency components
            threshold = np.max(magnitude) * 0.1
            significant_components = np.sum(magnitude > threshold)
            frequency_richness = min(significant_components / 1000.0, 1.0)
            assessment["frequency_richness"] = frequency_richness

            # Overall separation readiness score
            factors = [
                assessment.get("multi_instrument_likelihood", 0.5),
                assessment.get("dynamic_range_quality", 0.5),
                assessment.get("frequency_richness", 0.5)
            ]
            assessment["separation_readiness_score"] = np.mean(factors)

        except Exception as e:
            logger.warning(f"Failed to assess separation readiness: {e}")
            assessment["separation_readiness_score"] = 0.5

        return assessment

    def get_separation_recommendation(self, features: Dict) -> str:
        """Get recommendation for separation based on analysis."""
        readiness_score = features.get("separation_readiness_score", 0.5)

        if readiness_score >= 0.8:
            return "Excellent candidate for separation - high quality results expected"
        elif readiness_score >= 0.6:
            return "Good candidate for separation - should produce clean stems"
        elif readiness_score >= 0.4:
            return "Moderate candidate - separation possible but may have artifacts"
        else:
            return "Poor candidate - consider preprocessing or different approach"