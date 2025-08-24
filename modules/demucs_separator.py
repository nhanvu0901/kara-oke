import torch
import torchaudio
from pathlib import Path
from typing import Dict, Optional, Callable
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)
try:
    from demucs import pretrained
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
except ImportError:
    logger.warning("Demucs not installed. Install with: pip install demucs")
    pretrained = None


class DemucsSeparator:
    """Demucs v4 source separator for educational demonstration - Instrumental stems only."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", "cpu")
        self.model_name = config["demucs"]["model"]
        self.shifts = config["demucs"].get("shifts", 10)
        self.overlap = config["demucs"].get("overlap", 0.5)
        self.model = None

        if pretrained:
            self._load_model()

    def _load_model(self):
        """Load the Demucs model."""
        try:
            logger.info(f"Loading Demucs model: {self.model_name}")
            self.model = pretrained.get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Demucs model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Demucs model: {e}")
            raise

    def separate(self,
                 waveform: torch.Tensor,
                 sample_rate: int,
                 progress_callback: Optional[Callable] = None) -> Dict:
        """
        Separate audio into instrumental stems only (bass, drums, other).
        Vocals are excluded from processing and output.
        """
        if not self.model:
            raise RuntimeError("Demucs model not loaded")

        start_time = time.time()

        # Progress tracking
        if progress_callback:
            progress_callback(0.1, "Converting audio format...")

        try:
            # Prepare audio for Demucs
            audio = convert_audio(
                waveform.unsqueeze(0),
                sample_rate,
                self.model.samplerate,
                self.model.audio_channels
            )

            if progress_callback:
                progress_callback(0.2, "Audio converted, starting separation...")

            # Apply model with progress tracking
            class ProgressTracker:
                def __init__(self, callback):
                    self.callback = callback
                    self.current = 0.2

                def update(self, current, total):
                    if total > 0:
                        progress = 0.2 + (current / total) * 0.6  # 20% to 80%
                        if self.callback:
                            self.callback(progress, f"Processing chunk {current + 1}/{total}")

            tracker = ProgressTracker(progress_callback) if progress_callback else None

            with torch.no_grad():
                sources = apply_model(
                    self.model,
                    audio,
                    device=self.device,
                    shifts=min(self.shifts, 8),  # Limit shifts for reasonable speed
                    split=True,
                    overlap=self.overlap,
                    progress=tracker.update if tracker else False
                )

            if progress_callback:
                progress_callback(0.85, "Processing instrumental stems...")

            # Extract only instrumental stems (excluding vocals)
            stems = {}
            source_names = self.model.sources
            total_stems = len(source_names)

            # Filter out vocals from processing
            instrumental_sources = [name for name in source_names if name.lower() != 'vocals']

            for i, name in enumerate(source_names):
                # Skip vocals completely
                if name.lower() == 'vocals':
                    continue

                if progress_callback:
                    stem_progress = 0.85 + (i / total_stems) * 0.1
                    progress_callback(stem_progress, f"Processing {name} stem...")

                stem_audio = sources[0, i]

                # Convert back to original sample rate if needed
                if self.model.samplerate != sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        self.model.samplerate,
                        sample_rate
                    )
                    stem_audio = resampler(stem_audio)

                stems[name] = stem_audio

            if progress_callback:
                progress_callback(0.95, "Calculating quality metrics...")

            # Calculate metrics (excluding vocals)
            metrics = self._calculate_fast_metrics(waveform, stems)

            if progress_callback:
                progress_callback(1.0, "Separation complete!")

            processing_time = time.time() - start_time

            return {
                "stems": stems,
                "source_names": instrumental_sources,  # Only instrumental sources
                "metrics": metrics,
                "processing_time": processing_time,
                "model": self.model_name,
                "quality_settings": {
                    "shifts": min(self.shifts, 5),
                    "overlap": self.overlap
                },
                "note": "Instrumental stems only - vocals excluded"
            }

        except Exception as e:
            logger.error(f"Separation failed: {e}")
            raise

    def _calculate_fast_metrics(self, original: torch.Tensor, stems: Dict) -> Dict:
        """Fast calculation of essential metrics for instrumental stems only."""
        metrics = {}

        # Reconstruct from instrumental stems only
        reconstructed = torch.zeros_like(original)
        for stem in stems.values():
            if stem.shape != original.shape:
                if stem.shape[0] == 1 and original.shape[0] == 2:
                    stem = stem.repeat(2, 1)
                elif stem.shape[0] == 2 and original.shape[0] == 1:
                    stem = torch.mean(stem, dim=0, keepdim=True)
            reconstructed += stem

        # Note: SNR will be lower since we're not including vocals in reconstruction
        # This is expected and normal
        signal_power = torch.mean(original ** 2)
        instrumental_power = torch.mean(reconstructed ** 2)

        # Calculate instrumental-only metrics
        metrics["instrumental_energy_ratio"] = float(instrumental_power / (signal_power + 1e-10))
        metrics["num_instrumental_stems"] = len(stems)

        # Energy per instrumental stem
        for name, stem in stems.items():
            energy = torch.mean(stem ** 2)
            metrics[f"{name}_energy"] = float(energy)
            # Calculate relative energy among instrumentals
            metrics[f"{name}_instrumental_ratio"] = float(energy / (instrumental_power + 1e-10))

        return metrics

    def _calculate_detailed_metrics(self, original: torch.Tensor, stems: Dict) -> Dict:
        """Calculate comprehensive metrics for instrumental separation quality."""
        metrics = {}

        # Reconstruct from instrumental stems only
        reconstructed = torch.zeros_like(original)
        for stem in stems.values():
            if stem.shape != original.shape:
                # Handle channel mismatch
                if stem.shape[0] == 1 and original.shape[0] == 2:
                    stem = stem.repeat(2, 1)
                elif stem.shape[0] == 2 and original.shape[0] == 1:
                    stem = torch.mean(stem, dim=0, keepdim=True)
            reconstructed += stem

        # 1. Instrumental-only reconstruction metrics
        instrumental_power = torch.mean(reconstructed ** 2)
        original_power = torch.mean(original ** 2)

        # Instrumental coverage (how much of the instrumentals we captured)
        metrics["instrumental_coverage_ratio"] = float(instrumental_power / (original_power + 1e-10))

        # 2. Number of instrumental stems
        metrics["num_instrumental_stems"] = len(stems)

        # 3. Energy distribution across instrumental stems
        total_instrumental_energy = torch.sum(reconstructed ** 2)
        for name, stem in stems.items():
            energy = torch.sum(stem ** 2)
            energy_ratio = energy / (total_instrumental_energy + 1e-10)
            metrics[f"{name}_energy"] = float(energy)
            metrics[f"{name}_energy_ratio"] = float(energy_ratio)

        # 4. Dynamic range metrics per instrumental stem
        for name, stem in stems.items():
            if stem.numel() > 0:
                peak = torch.max(torch.abs(stem))
                rms = torch.sqrt(torch.mean(stem ** 2))
                dynamic_range = 20 * torch.log10(peak / (rms + 1e-10))
                metrics[f"{name}_dynamic_range_db"] = float(dynamic_range)
                metrics[f"{name}_peak_amplitude"] = float(peak)
                metrics[f"{name}_rms_energy"] = float(rms)

        # 5. Spectral metrics for instrumentals
        metrics.update(self._calculate_spectral_metrics(reconstructed, stems))

        # 6. Quality assessment for instrumental separation
        metrics["instrumental_quality_score"] = self._calculate_instrumental_quality_score(metrics)

        return metrics

    def _calculate_spectral_metrics(self, instrumental_mix: torch.Tensor, stems: Dict) -> Dict:
        """Calculate spectral-based quality metrics for instrumental stems."""
        spectral_metrics = {}

        # Convert to frequency domain
        mix_fft = torch.fft.rfft(instrumental_mix, dim=-1)
        mix_magnitude = torch.abs(mix_fft)

        # Analyze each instrumental stem
        for name, stem in stems.items():
            stem_fft = torch.fft.rfft(stem, dim=-1)
            stem_magnitude = torch.abs(stem_fft)

            # Spectral energy ratio for this stem
            stem_spectral_energy = torch.sum(stem_magnitude ** 2)
            mix_spectral_energy = torch.sum(mix_magnitude ** 2)
            spectral_metrics[f"{name}_spectral_ratio"] = float(
                stem_spectral_energy / (mix_spectral_energy + 1e-10)
            )

        # Frequency distribution quality
        sr = 44100  # Assume standard sample rate
        nyquist = sr // 2

        # Check low frequency content (bass region: 20-250 Hz)
        low_freq_bin = int(250 * mix_magnitude.shape[-1] / nyquist)
        low_freq_energy = torch.sum(mix_magnitude[..., :low_freq_bin] ** 2)

        # Check mid frequency content (250-4000 Hz)
        mid_freq_bin = int(4000 * mix_magnitude.shape[-1] / nyquist)
        mid_freq_energy = torch.sum(mix_magnitude[..., low_freq_bin:mid_freq_bin] ** 2)

        # Check high frequency content (4000+ Hz)
        high_freq_energy = torch.sum(mix_magnitude[..., mid_freq_bin:] ** 2)

        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
        spectral_metrics["low_freq_ratio"] = float(low_freq_energy / (total_energy + 1e-10))
        spectral_metrics["mid_freq_ratio"] = float(mid_freq_energy / (total_energy + 1e-10))
        spectral_metrics["high_freq_ratio"] = float(high_freq_energy / (total_energy + 1e-10))

        return spectral_metrics

    def _calculate_instrumental_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score for instrumental separation (0-100)."""
        # Factors for instrumental quality
        coverage = metrics.get("instrumental_coverage_ratio", 0)

        # Check balance between stems (ideal: relatively even distribution)
        energy_ratios = [v for k, v in metrics.items() if k.endswith("_energy_ratio")]
        if energy_ratios:
            balance_score = 1.0 - np.std(energy_ratios)  # Lower std = better balance
        else:
            balance_score = 0.5

        # Spectral distribution quality
        low_freq = metrics.get("low_freq_ratio", 0.3)
        mid_freq = metrics.get("mid_freq_ratio", 0.5)
        high_freq = metrics.get("high_freq_ratio", 0.2)

        # Ideal distribution: balanced across spectrum
        spectral_balance = 1.0 - abs(low_freq - 0.3) - abs(mid_freq - 0.5) - abs(high_freq - 0.2)

        # Weighted scoring for instrumental quality
        coverage_score = min(coverage * 1.2, 1.0) * 40  # 40% weight on coverage
        balance_score = balance_score * 30  # 30% weight on stem balance
        spectral_score = spectral_balance * 30  # 30% weight on spectral balance

        total_score = coverage_score + balance_score + spectral_score
        return min(total_score, 100.0)

    def get_quality_assessment(self, metrics: Dict) -> str:
        """Get human-readable quality assessment for instrumental separation."""
        quality_score = metrics.get("instrumental_quality_score", 0)
        coverage = metrics.get("instrumental_coverage_ratio", 0)

        if quality_score >= 80:
            return "Excellent - Clean instrumental separation with clear distinction between stems"
        elif quality_score >= 65:
            return "Very Good - Good instrumental separation with minor bleeding between stems"
        elif quality_score >= 50:
            return "Good - Acceptable instrumental separation for most applications"
        elif quality_score >= 35:
            return "Fair - Noticeable stem bleeding but usable for remixing"
        else:
            return "Poor - Significant artifacts in instrumental separation"