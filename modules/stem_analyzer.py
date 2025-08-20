# modules/stem_analyzer.py - COMPLETE IMPLEMENTATION
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("Librosa required for stem analysis")
    LIBROSA_AVAILABLE = False


class StemAnalyzer:
    """Intelligent stem analysis and classification."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def _load_instrument_profiles(self) -> Dict:
        """Define acoustic characteristics for instrument detection."""
        return {
            "drums": {
                "freq_range": (20, 8000),
                "onset_density_threshold": 0.7,  # High onset density
                "harmonic_ratio_threshold": 0.3,  # Low harmonic content
                "zero_crossing_rate": "high",
                "spectral_centroid_range": (1000, 5000)
            },
            "bass": {
                "freq_range": (30, 500),
                "onset_density_threshold": 0.4,
                "harmonic_ratio_threshold": 0.6,
                "fundamental_range": (30, 250),
                "spectral_centroid_range": (80, 400)
            },
            "vocals": {
                "freq_range": (80, 8000),
                "onset_density_threshold": 0.5,
                "harmonic_ratio_threshold": 0.8,
                "formant_presence": True,
                "spectral_centroid_range": (200, 3000)
            },
            "other": {
                "freq_range": (100, 10000),
                "onset_density_threshold": 0.5,
                "harmonic_ratio_threshold": 0.5,
                "spectral_centroid_range": (200, 5000)
            }
        }

    # In modules/stem_analyzer.py, SIMPLIFY the analyze_stem method:

    def analyze_stem(self, audio: torch.Tensor, stem_name: str) -> Dict:
        """Comprehensive stem analysis."""
        # Convert to numpy for librosa
        if audio.dim() > 1:
            audio_np = audio.mean(dim=0).numpy()
        else:
            audio_np = audio.numpy()

        # Extract all features
        features = {
            "name": stem_name,
            "instrument_type": stem_name,  # JUST USE THE STEM NAME AS TYPE!
            "energy": float(torch.mean(audio ** 2)),
        }

        if LIBROSA_AVAILABLE:
            # Extract audio features but NOT for detection
            features["frequency_content"] = self._analyze_frequency_content(audio_np)
            features["rhythmic_properties"] = self._analyze_rhythm(audio_np)
            features["harmonic_content"] = self._analyze_harmonics(audio_np)
            features["key"] = self._detect_key(audio_np)
            features["tempo_relevance"] = self._assess_tempo_relevance(audio_np)

        # Generate transformation recommendations based on stem name
        features["recommended_transformations"] = self._get_recommendations(features)

        return features



    def _analyze_frequency_content(self, audio: np.ndarray) -> Dict:
        """Analyze frequency content of audio."""
        # Compute spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)

        return {
            "spectral_centroid": float(np.mean(spectral_centroids)),
            "spectral_rolloff": float(np.mean(spectral_rolloff)),
            "spectral_bandwidth": float(np.mean(spectral_bandwidth)),
            "zero_crossing_rate": float(np.mean(zcr)),
            "brightness": float(np.mean(spectral_centroids) / 5000),  # Normalized brightness
        }

    def _analyze_rhythm(self, audio: np.ndarray) -> Dict:
        """Analyze rhythmic properties."""
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)

        # Onset density (onsets per second)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sample_rate)
        duration = len(audio) / self.sample_rate
        onset_density = len(onsets) / duration if duration > 0 else 0

        return {
            "tempo": float(tempo),
            "onset_density": onset_density,
            "beat_strength": float(np.mean(onset_env)),
            "rhythmic_complexity": float(np.std(onset_env)),
        }

    def _analyze_harmonics(self, audio: np.ndarray) -> Dict:
        """Analyze harmonic content."""
        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio)

        # Calculate energy ratios
        harmonic_energy = np.sum(harmonic ** 2)
        percussive_energy = np.sum(percussive ** 2)
        total_energy = np.sum(audio ** 2) + 1e-10

        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)

        return {
            "harmonic_ratio": harmonic_energy / total_energy,
            "percussive_ratio": percussive_energy / total_energy,
            "pitch_strength": float(np.mean(magnitudes)),
        }

    def _detect_key(self, audio: np.ndarray) -> str:
        """Detect musical key using chroma features."""
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate)

        # Average chroma vector
        chroma_mean = np.mean(chroma, axis=1)

        # Map to key (simplified - uses major keys only)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_idx = np.argmax(chroma_mean)

        return keys[key_idx]

    def _assess_tempo_relevance(self, audio: np.ndarray) -> float:
        """Assess how tempo-relevant the stem is."""
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)

        # Autocorrelation for tempo stability
        autocorr = librosa.autocorrelate(onset_env)

        # Find peaks in autocorrelation (indicates regular rhythm)
        peaks = librosa.util.peak_pick(autocorr, pre_max=3, post_max=3,
                                       pre_avg=3, post_avg=5, delta=0.5, wait=10)

        # More peaks = more regular rhythm = higher tempo relevance
        tempo_relevance = min(len(peaks) / 10.0, 1.0)

        return tempo_relevance

    def _get_recommendations(self, features: Dict) -> Dict:
        """Generate transformation recommendations based on stem type."""
        stem_name = features["name"].lower()  # Use the actual stem name
        recommendations = {}

        if "drum" in stem_name:
            recommendations = {
                "trap": {"description": "Modern trap drums with 808s", "compatibility": 0.9},
                "jazz": {"description": "Jazz drums with brushes", "compatibility": 0.7},
                "electronic": {"description": "Electronic dance drums", "compatibility": 0.85},
                "latin": {"description": "Latin percussion ensemble", "compatibility": 0.6},
            }
        elif "bass" in stem_name:
            recommendations = {
                "synth_bass": {"description": "Analog synthesizer bass", "compatibility": 0.8},
                "upright": {"description": "Acoustic upright bass", "compatibility": 0.7},
                "808": {"description": "808 sub bass", "compatibility": 0.85},
                "electric": {"description": "Electric bass guitar", "compatibility": 0.75},
            }
        elif "vocal" in stem_name:
            recommendations = {
                "saxophone": {"description": "Smooth saxophone melody", "compatibility": 0.7},
                "violin": {"description": "Expressive violin", "compatibility": 0.65},
                "synth_lead": {"description": "Synthesizer lead", "compatibility": 0.8},
                "choir": {"description": "Choir harmonies", "compatibility": 0.6},
            }
        elif "guitar" in stem_name:
            recommendations = {
                "acoustic": {"description": "Acoustic guitar", "compatibility": 0.8},
                "electric_clean": {"description": "Clean electric guitar", "compatibility": 0.75},
                "distorted": {"description": "Distorted rock guitar", "compatibility": 0.7},
                "jazz_guitar": {"description": "Smooth jazz guitar", "compatibility": 0.65},
            }
        elif "piano" in stem_name:
            recommendations = {
                "grand_piano": {"description": "Concert grand piano", "compatibility": 0.85},
                "electric_piano": {"description": "Electric piano (Rhodes)", "compatibility": 0.75},
                "synth_keys": {"description": "Synthesizer keys", "compatibility": 0.7},
                "harpsichord": {"description": "Baroque harpsichord", "compatibility": 0.6},
            }
        else:  # "other" or unknown
            recommendations = {
                "strings": {"description": "String ensemble", "compatibility": 0.7},
                "synth_pad": {"description": "Ambient synthesizer pad", "compatibility": 0.75},
                "brass": {"description": "Brass section", "compatibility": 0.65},
                "woodwinds": {"description": "Woodwind ensemble", "compatibility": 0.6},
            }

        return recommendations