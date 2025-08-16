import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Callable, Tuple
import logging
import time
import requests
import json

logger = logging.getLogger(__name__)

# Try to import demucs
try:
    from demucs import pretrained
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    DEMUCS_AVAILABLE = True
except ImportError:
    logger.warning("Demucs not installed. Will use fallback separation methods.")
    DEMUCS_AVAILABLE = False


class MVSEPSeparator:
    """Enhanced separator with multi-model support and intelligent processing."""

    # Available models and their capabilities
    MODELS = {
        "htdemucs": {"stems": 4, "quality": "high", "speed": "medium"},
        "htdemucs_ft": {"stems": 4, "quality": "very_high", "speed": "slow"},
        "htdemucs_6s": {"stems": 6, "quality": "very_high", "speed": "slow"},
        "mdx": {"stems": 4, "quality": "high", "speed": "fast"},
        "mdx_extra": {"stems": 4, "quality": "very_high", "speed": "medium"},
        "mdx_q": {"stems": 4, "quality": "medium", "speed": "very_fast"},
    }

    # Instrument-specific optimal models
    INSTRUMENT_MODELS = {
        "vocals": ["htdemucs_ft", "mdx_extra"],
        "drums": ["htdemucs", "mdx"],
        "bass": ["htdemucs_ft", "mdx_extra"],
        "piano": ["htdemucs_6s"],
        "guitar": ["htdemucs_6s"],
        "other": ["htdemucs_ft", "mdx"]
    }

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", "cpu")
        self.models_cache = {}
        self.use_demucs = DEMUCS_AVAILABLE and config.get("mvsep", {}).get("use_demucs", True)

        if self.use_demucs:
            self._preload_models()

    def _preload_models(self):
        """Preload commonly used models."""
        if self.use_demucs:
            try:
                # Load default model
                default_model = self.config.get("mvsep", {}).get("default_model", "htdemucs_ft")
                if "htdemucs" in default_model:
                    logger.info(f"Preloading Demucs model: {default_model}")
                    self.models_cache[default_model] = pretrained.get_model(default_model)
                    self.models_cache[default_model].to(self.device)
                    self.models_cache[default_model].eval()
            except Exception as e:
                logger.error(f"Failed to preload model: {e}")

    def separate(self,
                 waveform: torch.Tensor,
                 sample_rate: int,
                 target_instruments: Optional[List[str]] = None,
                 strategy: Optional[Dict] = None,
                 progress_callback: Optional[Callable] = None) -> Dict:
        """
        Multi-pass intelligent separation using Demucs backend.

        Args:
            waveform: Input audio tensor
            sample_rate: Sample rate
            target_instruments: Specific instruments to isolate
            strategy: LLM-recommended separation strategy
            progress_callback: Progress updates

        Returns:
            Dictionary with separated stems and quality metrics
        """
        start_time = time.time()

        # Determine separation strategy
        if strategy:
            models_to_use = strategy.get("models", ["htdemucs_ft"])
            num_passes = strategy.get("passes", 1)
            combination_method = strategy.get("combination_method", "average")
        else:
            models_to_use = [self.config.get("mvsep", {}).get("default_model", "htdemucs_ft")]
            num_passes = 1
            combination_method = "average"

        all_stems = {}
        pass_results = []

        # Multi-pass processing
        for pass_num in range(num_passes):
            if progress_callback:
                progress_callback(pass_num / num_passes, f"Pass {pass_num + 1}/{num_passes}")

            model_name = models_to_use[min(pass_num, len(models_to_use) - 1)]

            # Perform separation
            if self.use_demucs and "htdemucs" in model_name:
                stems = self._separate_with_demucs(waveform, sample_rate, model_name, progress_callback)
            else:
                stems = self._separate_with_frequency(waveform, sample_rate)

            # Calculate metrics for this pass
            metrics = self._calculate_separation_metrics(waveform, stems)

            pass_results.append({
                "model": model_name,
                "stems": stems,
                "metrics": metrics
            })

            # Merge or refine stems based on pass number
            if pass_num == 0:
                all_stems = stems
            else:
                all_stems = self._refine_stems(all_stems, stems, combination_method)

        # Post-process stems if target instruments specified
        if target_instruments:
            all_stems = self._extract_target_instruments(all_stems, target_instruments, waveform)

        # Apply quality enhancement
        if self.config.get("mvsep", {}).get("enhance_quality", True):
            all_stems = self._enhance_quality(all_stems, sample_rate)

        # Calculate final quality metrics
        final_metrics = self._calculate_final_metrics(waveform, all_stems, pass_results)

        return {
            "stems": all_stems,
            "passes": pass_results,
            "metrics": final_metrics,
            "processing_time": time.time() - start_time,
            "strategy_used": strategy
        }

    def _separate_with_demucs(self, waveform: torch.Tensor, sample_rate: int,
                              model_name: str, progress_callback: Optional[Callable] = None) -> Dict:
        """Separate using Demucs models."""
        try:
            # Get or load model
            if model_name not in self.models_cache:
                logger.info(f"Loading Demucs model: {model_name}")
                self.models_cache[model_name] = pretrained.get_model(model_name)
                self.models_cache[model_name].to(self.device)
                self.models_cache[model_name].eval()

            model = self.models_cache[model_name]

            # Prepare audio for Demucs
            audio = convert_audio(
                waveform.unsqueeze(0),  # Add batch dimension
                sample_rate,
                model.samplerate,
                model.audio_channels
            )

            # Apply model
            with torch.no_grad():
                sources = apply_model(
                    model,
                    audio.to(self.device),
                    device=self.device,
                    progress=True if progress_callback else False,
                    shifts=self.config.get("mvsep", {}).get("shifts", 1)
                )

            # Extract stems
            stems = {}
            source_names = model.sources

            for i, name in enumerate(source_names):
                stem_audio = sources[0, i].cpu()  # Remove batch dimension and move to CPU

                # Convert back to original sample rate if needed
                if model.samplerate != sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        model.samplerate,
                        sample_rate
                    )
                    stem_audio = resampler(stem_audio)

                stems[name] = stem_audio

            return stems

        except Exception as e:
            logger.error(f"Demucs separation failed: {e}")
            # Fallback to frequency-based separation
            return self._separate_with_frequency(waveform, sample_rate)

    def _separate_with_frequency(self, waveform: torch.Tensor, sample_rate: int) -> Dict:
        """Fallback frequency-based separation."""
        logger.info("Using frequency-based separation (fallback)")

        # Convert to mono if stereo
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            mono = waveform.mean(dim=0, keepdim=True)
        else:
            mono = waveform if waveform.dim() == 1 else waveform

        # Apply FFT
        fft = torch.fft.rfft(mono)
        freqs = torch.fft.rfftfreq(mono.shape[-1], 1 / sample_rate)

        stems = {}

        # Separate by frequency bands
        # Vocals: 85-3000 Hz (main vocal range)
        vocal_mask = (freqs >= 85) & (freqs <= 3000)
        vocals_fft = fft.clone()
        vocals_fft[~vocal_mask] *= 0.1  # Attenuate non-vocal frequencies
        stems["vocals"] = torch.fft.irfft(vocals_fft, n=mono.shape[-1])

        # Bass: 20-250 Hz
        bass_mask = (freqs >= 20) & (freqs <= 250)
        bass_fft = fft.clone()
        bass_fft[~bass_mask] *= 0.1
        stems["bass"] = torch.fft.irfft(bass_fft, n=mono.shape[-1])

        # Drums: Transients + 60-100 Hz (kick) + 5k-10k Hz (cymbals)
        drums_mask = ((freqs >= 60) & (freqs <= 100)) | ((freqs >= 5000) & (freqs <= 10000))
        drums_fft = fft.clone()
        drums_fft[~drums_mask] *= 0.3
        stems["drums"] = torch.fft.irfft(drums_fft, n=mono.shape[-1])

        # Other: Everything else
        other_mask = (freqs >= 250) & (freqs <= 8000)
        other_fft = fft.clone()
        other_fft[vocal_mask | bass_mask | drums_mask] *= 0.3
        stems["other"] = torch.fft.irfft(other_fft, n=mono.shape[-1])

        # Restore original shape
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            for name in stems:
                stems[name] = stems[name].repeat(waveform.shape[0], 1)

        return stems

    def _refine_stems(self, stems1: Dict, stems2: Dict, method: str = "average") -> Dict:
        """Intelligently combine stems from multiple passes."""
        refined = {}

        for stem_name in stems1.keys():
            if stem_name in stems2:
                if method == "average":
                    refined[stem_name] = (stems1[stem_name] + stems2[stem_name]) / 2
                elif method == "max_energy":
                    energy1 = torch.mean(stems1[stem_name] ** 2)
                    energy2 = torch.mean(stems2[stem_name] ** 2)
                    refined[stem_name] = stems1[stem_name] if energy1 > energy2 else stems2[stem_name]
                elif method == "frequency_split":
                    refined[stem_name] = self._frequency_combine(stems1[stem_name], stems2[stem_name])
                elif method == "quality_weighted":
                    # Weight by SNR or other quality metric
                    snr1 = self._calculate_snr(stems1[stem_name])
                    snr2 = self._calculate_snr(stems2[stem_name])
                    w1 = snr1 / (snr1 + snr2 + 1e-10)
                    w2 = snr2 / (snr1 + snr2 + 1e-10)
                    refined[stem_name] = w1 * stems1[stem_name] + w2 * stems2[stem_name]
                else:
                    refined[stem_name] = stems1[stem_name]
            else:
                refined[stem_name] = stems1[stem_name]

        # Add any stems from stems2 not in stems1
        for stem_name in stems2.keys():
            if stem_name not in refined:
                refined[stem_name] = stems2[stem_name]

        return refined

    def _frequency_combine(self, stem1: torch.Tensor, stem2: torch.Tensor) -> torch.Tensor:
        """Combine stems using frequency-based splitting."""
        fft1 = torch.fft.rfft(stem1)
        fft2 = torch.fft.rfft(stem2)

        # Use low frequencies from model 1, high from model 2
        split_point = len(fft1) // 3
        combined_fft = torch.cat([fft1[:split_point], fft2[split_point:]])

        return torch.fft.irfft(combined_fft, n=stem1.shape[-1])

    def _extract_target_instruments(self, stems: Dict, targets: List[str], original: torch.Tensor) -> Dict:
        """Extract specific instruments from stems."""
        extracted = {}

        for target in targets:
            if target in stems:
                extracted[target] = stems[target]
            elif target in ["piano", "guitar", "strings", "synth"] and "other" in stems:
                # Try to extract from "other" stem using targeted filtering
                extracted[target] = self._extract_from_other(stems["other"], target, original)
            else:
                logger.warning(f"Target instrument '{target}' not found in stems")

        return extracted

    def _extract_from_other(self, other_stem: torch.Tensor, instrument: str,
                            original: torch.Tensor) -> torch.Tensor:
        """Extract specific instrument from 'other' stem using targeted processing."""
        # Apply instrument-specific filtering based on typical frequency ranges
        instrument_ranges = {
            "piano": (27.5, 4186.0),
            "guitar": (82.4, 1318.5),
            "strings": (196.0, 3520.0),
            "synth": (20.0, 20000.0)
        }

        if instrument in instrument_ranges:
            low_freq, high_freq = instrument_ranges[instrument]

            # Apply bandpass filter
            sample_rate = self.config.get("sample_rate", 44100)
            fft = torch.fft.rfft(other_stem)
            freqs = torch.fft.rfftfreq(other_stem.shape[-1], 1 / sample_rate)

            mask = (freqs >= low_freq) & (freqs <= high_freq)
            fft[~mask] *= 0.1  # Attenuate out-of-range frequencies

            return torch.fft.irfft(fft, n=other_stem.shape[-1])

        return other_stem

    def _enhance_quality(self, stems: Dict, sample_rate: int) -> Dict:
        """Apply quality enhancement to separated stems."""
        enhanced = {}

        for name, stem in stems.items():
            # Remove DC offset
            stem = stem - torch.mean(stem)

            # Apply gentle noise gate to reduce artifacts
            threshold = torch.max(torch.abs(stem)) * 0.01
            stem[torch.abs(stem) < threshold] *= 0.1

            # Normalize without clipping
            max_val = torch.max(torch.abs(stem))
            if max_val > 0:
                stem = stem * (0.95 / max_val)

            enhanced[name] = stem

        return enhanced

    def _calculate_snr(self, signal: torch.Tensor) -> float:
        """Calculate signal-to-noise ratio."""
        # Simple SNR estimation
        signal_power = torch.mean(signal ** 2)
        # Estimate noise as the power in quiet parts
        sorted_power = torch.sort(signal ** 2)[0]
        noise_power = torch.mean(sorted_power[:len(sorted_power) // 10])  # Bottom 10%

        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        return float(snr)

    def _calculate_separation_metrics(self, original: torch.Tensor, stems: Dict) -> Dict:
        """Calculate quality metrics for separation."""
        metrics = {}

        # Reconstruction quality
        reconstructed = sum(stems.values())
        if reconstructed.shape != original.shape:
            # Handle shape mismatch
            if original.dim() > reconstructed.dim():
                reconstructed = reconstructed.unsqueeze(0).repeat(original.shape[0], 1)

        mse = torch.mean((original - reconstructed) ** 2)
        signal_power = torch.mean(original ** 2)
        snr = 10 * torch.log10(signal_power / (mse + 1e-10))
        metrics["reconstruction_snr"] = float(snr)

        # Cross-talk estimation (simplified)
        for name, stem in stems.items():
            metrics[f"{name}_energy"] = float(torch.mean(stem ** 2))
            metrics[f"{name}_snr"] = self._calculate_snr(stem)

        return metrics

    def _calculate_final_metrics(self, original: torch.Tensor, stems: Dict, passes: List) -> Dict:
        """Calculate comprehensive final metrics."""
        metrics = {
            "num_stems": len(stems),
            "num_passes": len(passes),
            "models_used": [p["model"] for p in passes]
        }

        # Overall quality score
        reconstruction_snr = self._calculate_separation_metrics(original, stems)["reconstruction_snr"]
        metrics["quality_score"] = min(100, reconstruction_snr * 2.5)  # Normalized to 0-100

        # Best pass metrics
        best_snr = max(p["metrics"]["reconstruction_snr"] for p in passes)
        metrics["best_pass_snr"] = best_snr

        return metrics

    def recommend_strategy(self, features: Dict, target_quality: str = "high") -> Dict:
        """Recommend separation strategy based on audio features."""
        strategy = {
            "models": [],
            "passes": 1,
            "combination_method": "average"
        }

        # Analyze features to determine best approach
        has_vocals = features.get("vocal_presence", 0) > 0.3
        is_complex = features.get("spectral_complexity", 0) > 0.7
        has_percussion = features.get("percussive_ratio", 0) > 0.2

        if target_quality == "very_high" and is_complex:
            strategy["models"] = ["htdemucs_ft", "htdemucs_6s"]
            strategy["passes"] = 2
            strategy["combination_method"] = "quality_weighted"
        elif has_vocals and has_percussion:
            strategy["models"] = ["htdemucs_ft"]
            strategy["passes"] = 1
            strategy["combination_method"] = "average"
        else:
            strategy["models"] = ["htdemucs"]
            strategy["passes"] = 1

        # Add shifts for better quality
        if target_quality in ["high", "very_high"]:
            strategy["shifts"] = 5 if target_quality == "very_high" else 2

        return strategy