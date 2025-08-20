# modules/stem_transformer.py - COMPLETE IMPLEMENTATION
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import time

logger = logging.getLogger(__name__)


class StemTransformer:
    """Intelligent stem transformation with style transfer."""

    def __init__(self, config: Dict, audiocraft_processor=None):
        self.config = config
        self.device = config.get("device", "cpu")
        self.audiocraft = audiocraft_processor
        self.transformation_presets = self._load_presets()
        self.model_cache = {}

    def _load_presets(self) -> Dict:
        """Load transformation presets for different instrument types."""
        return {
            "drums": {
                "trap": {"prompt": "trap drums with deep 808 bass kicks, crisp hi-hats, snappy snares",
                         "temperature": 0.8},
                "jazz": {"prompt": "jazz drums with soft brushes on snare, subtle ride cymbal", "temperature": 0.7},
                "electronic": {"prompt": "electronic dance drums with punchy kicks and synthetic percussion",
                               "temperature": 0.9},
                "latin": {"prompt": "latin percussion with congas, timbales, and shakers", "temperature": 0.75},
            },
            "bass": {
                "synth_bass": {"prompt": "warm analog synthesizer bass with filter resonance", "temperature": 0.7},
                "upright": {"prompt": "acoustic upright bass with natural wood tone", "temperature": 0.6},
                "808": {"prompt": "deep 808 sub bass with long sustain", "temperature": 0.8},
                "electric": {"prompt": "electric bass guitar with round warm tone", "temperature": 0.65},
            },
            "vocals": {
                "saxophone": {"prompt": "smooth jazz tenor saxophone solo", "temperature": 0.6},
                "violin": {"prompt": "expressive solo violin with vibrato", "temperature": 0.7},
                "synth_lead": {"prompt": "analog synthesizer lead with portamento glide", "temperature": 0.8},
                "choir": {"prompt": "ethereal choir vocals with harmonies", "temperature": 0.65},
            },
            "other": {
                "piano": {"prompt": "grand piano with rich harmonics", "temperature": 0.6},
                "strings": {"prompt": "orchestral string section", "temperature": 0.7},
                "synth_pad": {"prompt": "ambient synthesizer pad with slow evolution", "temperature": 0.8},
                "guitar": {"prompt": "electric guitar with clean tone", "temperature": 0.65},
            }
        }

    def transform_stem(self,
                       audio: torch.Tensor,
                       stem_info: Dict,
                       transformation: str,
                       progress_callback: Optional[Callable] = None) -> Dict:
        """Transform a single stem with selected style."""

        start_time = time.time()

        # Use stem name as instrument type
        stem_name = stem_info.get("name", "other")
        instrument_type = stem_name.lower()  # Direct mapping!

        # Get preset based on stem name
        if instrument_type in self.transformation_presets and transformation in self.transformation_presets[
            instrument_type]:
            preset = self.transformation_presets[instrument_type][transformation]
        else:
            # Try to find in any category
            preset = None
            for category, transforms in self.transformation_presets.items():
                if transformation in transforms:
                    preset = transforms[transformation]
                    break

            if not preset:
                preset = {"prompt": f"transform to {transformation}", "temperature": 0.7}

        if progress_callback:
            progress_callback(0.1, f"Preparing {transformation} transformation...")

        # Apply transformation using AudioCraft
        if self.audiocraft:
            if progress_callback:
                progress_callback(0.3, "Applying MusicGen transformation...")

            transformed_result = self.audiocraft.process(
                audio,
                stem_info.get("sample_rate", 44100),
                mode="continue",
                prompt=preset["prompt"]
            )

            transformed_audio = transformed_result["audio"]

            if progress_callback:
                progress_callback(0.8, "Assessing quality...")
        else:
            # Fallback: return original if AudioCraft not available
            logger.warning("AudioCraft not available, returning original audio")
            transformed_audio = audio

        # Quality assessment
        quality_score = self._assess_transformation_quality(audio, transformed_audio)

        # Check harmonic compatibility
        original_key = stem_info.get("key", "C")
        harmonic_compatibility = self._check_harmonic_compatibility(transformed_audio, original_key)

        if progress_callback:
            progress_callback(1.0, "Transformation complete!")

        processing_time = time.time() - start_time

        return {
            "audio": transformed_audio,
            "transformation": transformation,
            "quality_score": quality_score,
            "harmonic_compatibility": harmonic_compatibility,
            "processing_time": processing_time,
            "preset_used": preset
        }

    def batch_transform(self,
                        stems: Dict[str, torch.Tensor],
                        stem_analyses: Dict[str, Dict],
                        transformations: Dict[str, str],
                        parallel: bool = True) -> Dict:
        """Transform multiple stems with optimizations."""

        transformed_stems = {}

        if parallel and len(stems) > 1:
            # Parallel processing
            max_workers = min(self.config.get("max_workers", 4), len(stems))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}

                for stem_name, audio in stems.items():
                    if stem_name in transformations:
                        stem_info = stem_analyses.get(stem_name, {})
                        stem_info["sample_rate"] = self.config.get("sample_rate", 44100)

                        future = executor.submit(
                            self.transform_stem,
                            audio,
                            stem_info,
                            transformations[stem_name]
                        )
                        futures[future] = stem_name

                # Collect results
                for future in as_completed(futures):
                    stem_name = futures[future]
                    try:
                        result = future.result()
                        transformed_stems[stem_name] = result
                        logger.info(f"Transformed {stem_name}: quality={result['quality_score']:.2f}")
                    except Exception as e:
                        logger.error(f"Failed to transform {stem_name}: {e}")
                        # Keep original on failure
                        transformed_stems[stem_name] = {
                            "audio": stems[stem_name],
                            "transformation": "original",
                            "quality_score": 1.0,
                            "error": str(e)
                        }
        else:
            # Sequential processing
            for stem_name, audio in stems.items():
                if stem_name in transformations:
                    stem_info = stem_analyses.get(stem_name, {})
                    stem_info["sample_rate"] = self.config.get("sample_rate", 44100)

                    result = self.transform_stem(
                        audio,
                        stem_info,
                        transformations[stem_name]
                    )
                    transformed_stems[stem_name] = result

        return transformed_stems

    def _assess_transformation_quality(self, original: torch.Tensor, transformed: torch.Tensor) -> float:
        """Assess quality of transformation."""

        # Energy preservation ratio
        energy_ratio = self._calculate_energy_ratio(original, transformed)

        # Spectral similarity
        spectral_similarity = self._calculate_spectral_correlation(original, transformed)

        # Transient preservation
        transient_quality = self._assess_transient_quality(original, transformed)

        # Overall quality score (weighted average)
        quality_score = (
                energy_ratio * 0.2 +
                spectral_similarity * 0.4 +
                transient_quality * 0.4
        )

        return float(min(quality_score, 1.0))

    def _calculate_energy_ratio(self, original: torch.Tensor, transformed: torch.Tensor) -> float:
        """Calculate energy preservation ratio."""
        original_energy = torch.mean(original ** 2)
        transformed_energy = torch.mean(transformed ** 2)

        # Ideal ratio is close to 1.0
        ratio = transformed_energy / (original_energy + 1e-10)

        # Convert to quality score (1.0 is perfect)
        if ratio > 1:
            score = 1.0 / ratio  # Penalize over-amplification
        else:
            score = ratio  # Penalize energy loss

        return float(score)

    def _calculate_spectral_correlation(self, original: torch.Tensor, transformed: torch.Tensor) -> float:
        """Calculate spectral correlation between original and transformed."""
        # Compute spectrograms
        n_fft = 2048

        # Ensure same shape
        min_len = min(original.shape[-1], transformed.shape[-1])
        original = original[..., :min_len]
        transformed = transformed[..., :min_len]

        # FFT
        original_fft = torch.fft.rfft(original.flatten(), n=n_fft)
        transformed_fft = torch.fft.rfft(transformed.flatten(), n=n_fft)

        # Magnitude spectra
        original_mag = torch.abs(original_fft)
        transformed_mag = torch.abs(transformed_fft)

        # Correlation
        correlation = torch.corrcoef(torch.stack([
            original_mag[:len(original_mag) // 2],  # Focus on important frequencies
            transformed_mag[:len(transformed_mag) // 2]
        ]))[0, 1]

        return float(max(0, correlation))  # Ensure non-negative

    def _assess_transient_quality(self, original: torch.Tensor, transformed: torch.Tensor) -> float:
        """Assess how well transients are preserved."""
        # Simple onset detection using energy differences

        # Compute energy envelopes
        window_size = 2048
        hop_size = 512

        original_envelope = self._compute_energy_envelope(original, window_size, hop_size)
        transformed_envelope = self._compute_energy_envelope(transformed, window_size, hop_size)

        # Normalize envelopes
        original_envelope = original_envelope / (torch.max(original_envelope) + 1e-10)
        transformed_envelope = transformed_envelope / (torch.max(transformed_envelope) + 1e-10)

        # Compare onset patterns (simplified)
        correlation = torch.corrcoef(torch.stack([
            original_envelope,
            transformed_envelope[:len(original_envelope)]
        ]))[0, 1]

        return float(max(0, correlation))

    def _compute_energy_envelope(self, audio: torch.Tensor, window_size: int, hop_size: int) -> torch.Tensor:
        """Compute energy envelope of audio."""
        audio_flat = audio.flatten()
        num_frames = (len(audio_flat) - window_size) // hop_size + 1

        envelope = torch.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_size
            end = start + window_size
            envelope[i] = torch.mean(audio_flat[start:end] ** 2)

        return envelope

    def _check_harmonic_compatibility(self, audio: torch.Tensor, original_key: str) -> float:
        """Check if transformed audio maintains harmonic compatibility."""
        # Simplified key detection using chroma features

        # Compute chromagram using FFT
        n_fft = 4096
        audio_flat = audio.flatten()

        # FFT
        fft = torch.fft.rfft(audio_flat, n=n_fft)
        magnitude = torch.abs(fft)

        # Map to 12 chroma bins (simplified)
        chroma = torch.zeros(12)
        freqs = torch.fft.rfftfreq(n_fft, 1 / 44100.0)

        # A4 = 440 Hz
        for i, freq in enumerate(freqs):
            if freq > 0:
                # Map frequency to chroma bin
                pitch = 12 * torch.log2(freq / 440.0) + 69
                chroma_bin = int(pitch) % 12
                if 0 <= chroma_bin < 12:
                    chroma[chroma_bin] += magnitude[i]

        # Normalize chroma
        chroma = chroma / (torch.sum(chroma) + 1e-10)

        # Detect key (simplified - just find max chroma)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key_idx = torch.argmax(chroma)
        detected_key = keys[detected_key_idx]

        # Calculate compatibility
        compatibility = self._calculate_key_compatibility(original_key, detected_key)

        return compatibility

    def _calculate_key_compatibility(self, key1: str, key2: str) -> float:
        """Calculate compatibility between two musical keys."""
        # Circle of fifths distance
        keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']

        # Simplified: if same key, perfect compatibility
        if key1 == key2:
            return 1.0

        # Check if keys are in the circle
        if key1 in keys and key2 in keys:
            idx1 = keys.index(key1)
            idx2 = keys.index(key2)

            # Calculate circular distance
            distance = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))

            # Convert to compatibility (closer = more compatible)
            compatibility = 1.0 - (distance / 6.0)  # Max distance is 6

            return max(0.3, compatibility)  # Minimum compatibility of 0.3

        # Default compatibility for unknown keys
        return 0.5