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

            # Instrument-specific features
            features.update(self.extract_instrument_features(audio_np, sample_rate))

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

            # Enhanced features for instrument classification
            # Harmonic-to-noise ratio
            harmonic, percussive = librosa.effects.hpss(audio)
            features["harmonic_ratio"] = float(np.sum(harmonic ** 2) / (np.sum(audio ** 2) + 1e-10))
            features["percussive_ratio"] = float(np.sum(percussive ** 2) / (np.sum(audio ** 2) + 1e-10))

            # Spectral complexity
            spectral_entropy = -np.sum(centroid * np.log(centroid + 1e-10))
            features["spectral_complexity"] = float(spectral_entropy / (np.log(len(centroid)) + 1e-10))

            # Frequency band energy distribution
            bands = [(20, 250), (250, 500), (500, 2000), (2000, 4000), (4000, 8000), (8000, 20000)]
            stft = librosa.stft(audio)
            freqs = librosa.fft_frequencies(sr=sr)

            for i, (low, high) in enumerate(bands):
                band_mask = (freqs >= low) & (freqs <= high)
                band_energy = np.sum(np.abs(stft[band_mask, :]) ** 2)
                features[f"band_{i}_energy"] = float(band_energy)

            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features["spectral_contrast_mean"] = float(np.mean(contrast))
            features["spectral_contrast_std"] = float(np.std(contrast))

            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
            features["spectral_bandwidth_std"] = float(np.std(bandwidth))

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
            features["onset_strength_std"] = float(np.std(onset_env))

            # Tempo stability
            if len(beats) > 1:
                beat_times = librosa.frames_to_time(beats, sr=sr)
                beat_intervals = np.diff(beat_times)
                features["tempo_stability"] = float(1.0 / (np.std(beat_intervals) + 1e-10))
            else:
                features["tempo_stability"] = 0.0

            # Rhythmic complexity
            tempogram = librosa.feature.tempogram(y=audio, sr=sr)
            features["rhythmic_complexity"] = float(np.mean(np.std(tempogram, axis=1)))

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

            # Harmonic content analysis
            chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr)
            features["chroma_energy_mean"] = float(np.mean(chroma))
            features["chroma_energy_std"] = float(np.std(chroma))

            # Key estimation
            chroma_mean = np.mean(chroma, axis=1)
            key_index = np.argmax(chroma_mean)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            features["estimated_key"] = keys[key_index]
            features["key_confidence"] = float(chroma_mean[key_index])

        except Exception as e:
            logger.warning(f"Failed to extract harmonic features: {e}")

        return features

    def extract_instrument_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract features for instrument classification."""
        features = {}

        try:
            # Extract pitches and magnitudes first (needed for multiple analyses below)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)

            # Attack/decay characteristics
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3,
                                           pre_avg=3, post_avg=5, delta=0.5, wait=10)

            if len(peaks) > 0:
                # Attack time estimation
                attack_times = []
                for peak in peaks[:10]:  # Analyze first 10 onsets
                    if peak > 0:
                        attack = onset_env[peak] - onset_env[peak - 1]
                        attack_times.append(attack)
                features["attack_sharpness"] = float(np.mean(attack_times)) if attack_times else 0

                # Decay characteristics
                decay_rates = []
                for peak in peaks[:10]:
                    if peak < len(onset_env) - 10:
                        decay_segment = onset_env[peak:peak + 10]
                        decay_rate = np.mean(np.diff(decay_segment))
                        decay_rates.append(abs(decay_rate))
                features["decay_rate"] = float(np.mean(decay_rates)) if decay_rates else 0
            else:
                features["attack_sharpness"] = 0
                features["decay_rate"] = 0

            # Timbre descriptors
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features["tonnetz_mean"] = float(np.mean(tonnetz))
            features["tonnetz_std"] = float(np.std(tonnetz))

            # Chroma features for harmonic content
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features["chroma_mean"] = float(np.mean(chroma))
            features["chroma_std"] = float(np.std(chroma))

            # Onset patterns
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            features["onset_density"] = len(onset_frames) / (len(audio) / sr)

            # Onset regularity (for distinguishing rhythmic vs melodic instruments)
            if len(onset_frames) > 2:
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                onset_intervals = np.diff(onset_times)
                features["onset_regularity"] = float(1.0 / (np.std(onset_intervals) + 1e-10))
            else:
                features["onset_regularity"] = 0

            # Vocal presence detection
            S = np.abs(librosa.stft(audio))
            freqs = librosa.fft_frequencies(sr=sr)

            # Multiple vocal indicators
            # 1. Main vocal range energy
            vocal_range = (85, 3000)  # Typical vocal range
            vocal_mask = (freqs >= vocal_range[0]) & (freqs <= vocal_range[1])
            vocal_energy = np.sum(S[vocal_mask, :])
            total_energy = np.sum(S)
            features["vocal_presence"] = float(vocal_energy / (total_energy + 1e-10))

            # 2. Formant detection (simplified)
            formant_ranges = [(700, 1220), (1220, 2600)]  # F1 and F2 ranges
            formant_energy = 0
            for low, high in formant_ranges:
                formant_mask = (freqs >= low) & (freqs <= high)
                formant_energy += np.sum(S[formant_mask, :])
            features["formant_presence"] = float(formant_energy / (total_energy + 1e-10))

            # Instrument-specific frequency signatures
            # Piano characteristics
            piano_harmonics = [27.5, 55, 110, 220, 440, 880, 1760, 3520]
            piano_score = 0
            for harm in piano_harmonics:
                harm_idx = np.argmin(np.abs(freqs - harm))
                if harm_idx < len(S):
                    piano_score += np.mean(S[harm_idx, :])
            features["piano_likelihood"] = float(piano_score / (len(piano_harmonics) * np.mean(S) + 1e-10))

            # Guitar characteristics (fundamental + harmonics pattern)
            guitar_fundamentals = [82.4, 110, 146.8, 196, 246.9, 329.6]
            guitar_score = 0
            for fund in guitar_fundamentals:
                fund_idx = np.argmin(np.abs(freqs - fund))
                if fund_idx < len(S):
                    guitar_score += np.mean(S[fund_idx, :])
            features["guitar_likelihood"] = float(guitar_score / (len(guitar_fundamentals) * np.mean(S) + 1e-10))

            # Drum characteristics (broadband noise + transients)
            drum_bands = [(20, 100), (100, 200), (5000, 10000)]
            drum_score = 0
            for low, high in drum_bands:
                drum_mask = (freqs >= low) & (freqs <= high)
                drum_score += np.sum(S[drum_mask, :])
            features["drum_likelihood"] = float(drum_score / (total_energy + 1e-10))

            # Extract fundamental candidates for harmonic analysis
            fundamental_candidates = pitches[magnitudes > np.max(magnitudes) * 0.5]

            # String instruments (violin, cello, etc.)
            # Rich harmonic content with specific patterns
            string_harmonics_ratio = []
            for fund in fundamental_candidates[:5]:
                if fund > 0:
                    harmonic_power = 0
                    for n in range(2, 8):  # Check harmonics 2-7
                        harm_freq = fund * n
                        harm_idx = np.argmin(np.abs(freqs - harm_freq))
                        if harm_idx < len(S):
                            harmonic_power += np.mean(S[harm_idx, :])
                    string_harmonics_ratio.append(harmonic_power)
            features["strings_likelihood"] = float(np.mean(string_harmonics_ratio)) if string_harmonics_ratio else 0

            # Synthesizer detection (very stable pitch, less natural harmonics)
            valid_pitches = pitches[magnitudes > np.max(magnitudes) * 0.3]
            if len(valid_pitches[valid_pitches > 0]) > 0:
                pitch_stability = np.std(valid_pitches[valid_pitches > 0])
                features["synth_likelihood"] = float(1.0 / (pitch_stability + 1e-10))
            else:
                features["synth_likelihood"] = 0

            # Brass characteristics (strong odd harmonics)
            brass_score = 0
            if len(fundamental_candidates) > 0 and fundamental_candidates[0] > 0:
                fund = fundamental_candidates[0]
                for n in [1, 3, 5, 7]:  # Odd harmonics
                    harm_freq = fund * n
                    harm_idx = np.argmin(np.abs(freqs - harm_freq))
                    if harm_idx < len(S):
                        brass_score += np.mean(S[harm_idx, :])
            features["brass_likelihood"] = float(brass_score / (4 * np.mean(S) + 1e-10))

        except Exception as e:
            logger.warning(f"Failed to extract instrument features: {e}")

        return features

    def analyze_separation_quality(self, original: torch.Tensor, separated_stems: Dict[str, torch.Tensor]) -> Dict:
        """Analyze the quality of source separation."""
        metrics = {}

        try:
            # Convert to numpy
            if original.dim() > 1:
                original_np = original.mean(dim=0).numpy()
            else:
                original_np = original.numpy()

            # Reconstruction error
            reconstructed = torch.zeros_like(original)
            for stem in separated_stems.values():
                reconstructed += stem

            mse = torch.mean((original - reconstructed) ** 2)
            metrics["reconstruction_mse"] = float(mse)

            # Signal-to-noise ratio
            signal_power = torch.mean(original ** 2)
            noise_power = mse
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
            metrics["reconstruction_snr_db"] = float(snr)

            # Cross-talk estimation between stems
            cross_talk = {}
            stem_names = list(separated_stems.keys())
            for i, name1 in enumerate(stem_names):
                for name2 in stem_names[i + 1:]:
                    correlation = torch.corrcoef(torch.stack([
                        separated_stems[name1].flatten(),
                        separated_stems[name2].flatten()
                    ]))[0, 1]
                    cross_talk[f"{name1}_{name2}_correlation"] = float(correlation)
            metrics["cross_talk"] = cross_talk

            # Frequency distribution preservation
            if LIBROSA_AVAILABLE:
                orig_stft = librosa.stft(original_np)
                orig_mag = np.abs(orig_stft)

                recon_np = reconstructed.mean(dim=0).numpy() if reconstructed.dim() > 1 else reconstructed.numpy()
                recon_stft = librosa.stft(recon_np)
                recon_mag = np.abs(recon_stft)

                # Spectral distortion
                spectral_distortion = np.mean(np.abs(orig_mag - recon_mag))
                metrics["spectral_distortion"] = float(spectral_distortion)

                # Per-stem quality metrics
                for name, stem in separated_stems.items():
                    stem_np = stem.mean(dim=0).numpy() if stem.dim() > 1 else stem.numpy()

                    # Energy ratio
                    stem_energy = np.sum(stem_np ** 2)
                    total_energy = np.sum(original_np ** 2)
                    metrics[f"{name}_energy_ratio"] = float(stem_energy / (total_energy + 1e-10))

                    # Spectral clarity (ratio of harmonic to noise)
                    harmonic, _ = librosa.effects.hpss(stem_np)
                    clarity = np.sum(harmonic ** 2) / (np.sum(stem_np ** 2) + 1e-10)
                    metrics[f"{name}_clarity"] = float(clarity)

        except Exception as e:
            logger.warning(f"Failed to analyze separation quality: {e}")

        return metrics