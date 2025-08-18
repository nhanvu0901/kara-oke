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
    """Demucs v4 source separator for educational demonstration."""

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
        Separate audio into stems using Demucs v4 with progress tracking.
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
                progress_callback(0.85, "Processing stems...")

            # Extract stems
            stems = {}
            source_names = self.model.sources
            total_stems = len(source_names)

            for i, name in enumerate(source_names):
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

            # Calculate metrics (simplified for speed)
            metrics = self._calculate_fast_metrics(waveform, stems)

            if progress_callback:
                progress_callback(1.0, "Separation complete!")

            processing_time = time.time() - start_time

            return {
                "stems": stems,
                "source_names": source_names,
                "metrics": metrics,
                "processing_time": processing_time,
                "model": self.model_name,
                "quality_settings": {
                    "shifts": min(self.shifts, 5),
                    "overlap": self.overlap
                }
            }

        except Exception as e:
            logger.error(f"Separation failed: {e}")
            raise

    def _calculate_fast_metrics(self, original: torch.Tensor, stems: Dict) -> Dict:
        """Fast calculation of essential metrics only."""
        metrics = {}

        # Reconstruct from stems
        reconstructed = torch.zeros_like(original)
        for stem in stems.values():
            if stem.shape != original.shape:
                if stem.shape[0] == 1 and original.shape[0] == 2:
                    stem = stem.repeat(2, 1)
                elif stem.shape[0] == 2 and original.shape[0] == 1:
                    stem = torch.mean(stem, dim=0, keepdim=True)
            reconstructed += stem

        # Essential metrics only
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean((original - reconstructed) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        metrics["reconstruction_snr"] = float(snr)
        metrics["num_stems"] = len(stems)

        # Energy per stem
        for name, stem in stems.items():
            energy = torch.mean(stem ** 2)
            metrics[f"{name}_energy"] = float(energy)

        return metrics

    def _calculate_detailed_metrics(self, original: torch.Tensor, stems: Dict) -> Dict:
        """Calculate comprehensive metrics for separation quality assessment."""
        metrics = {}

        # Reconstruct from stems
        reconstructed = torch.zeros_like(original)
        for stem in stems.values():
            if stem.shape != original.shape:
                # Handle channel mismatch
                if stem.shape[0] == 1 and original.shape[0] == 2:
                    stem = stem.repeat(2, 1)
                elif stem.shape[0] == 2 and original.shape[0] == 1:
                    stem = torch.mean(stem, dim=0, keepdim=True)
            reconstructed += stem

        # 1. Signal-to-Noise Ratio (SNR) - Primary quality metric
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean((original - reconstructed) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        metrics["reconstruction_snr"] = float(snr)

        # 2. Source-to-Distortion Ratio (SDR) approximation
        distortion = original - reconstructed
        sdr = 10 * torch.log10(signal_power / (torch.mean(distortion ** 2) + 1e-10))
        metrics["source_distortion_ratio"] = float(sdr)

        # 3. Number of stems
        metrics["num_stems"] = len(stems)

        # 4. Energy distribution across stems
        total_energy = torch.sum(original ** 2)
        for name, stem in stems.items():
            energy = torch.sum(stem ** 2)
            energy_ratio = energy / (total_energy + 1e-10)
            metrics[f"{name}_energy"] = float(energy)
            metrics[f"{name}_energy_ratio"] = float(energy_ratio)

        # 5. Dynamic range metrics per stem
        for name, stem in stems.items():
            if stem.numel() > 0:
                peak = torch.max(torch.abs(stem))
                rms = torch.sqrt(torch.mean(stem ** 2))
                dynamic_range = 20 * torch.log10(peak / (rms + 1e-10))
                metrics[f"{name}_dynamic_range_db"] = float(dynamic_range)
                metrics[f"{name}_peak_amplitude"] = float(peak)
                metrics[f"{name}_rms_energy"] = float(rms)

        # 6. Spectral metrics
        metrics.update(self._calculate_spectral_metrics(original, stems))

        # 7. Quality assessment
        metrics["overall_quality_score"] = self._calculate_quality_score(metrics)

        return metrics

    def _calculate_spectral_metrics(self, original: torch.Tensor, stems: Dict) -> Dict:
        """Calculate spectral-based quality metrics."""
        spectral_metrics = {}

        # Convert to frequency domain
        original_fft = torch.fft.rfft(original, dim=-1)
        original_magnitude = torch.abs(original_fft)

        # Reconstruct in frequency domain
        reconstructed_fft = torch.zeros_like(original_fft)
        for stem in stems.values():
            if stem.shape == original.shape:
                stem_fft = torch.fft.rfft(stem, dim=-1)
                reconstructed_fft += stem_fft

        reconstructed_magnitude = torch.abs(reconstructed_fft)

        # Spectral SNR
        spectral_error = torch.abs(original_magnitude - reconstructed_magnitude)
        spectral_signal_power = torch.mean(original_magnitude ** 2)
        spectral_noise_power = torch.mean(spectral_error ** 2)
        spectral_snr = 10 * torch.log10(spectral_signal_power / (spectral_noise_power + 1e-10))
        spectral_metrics["spectral_snr"] = float(spectral_snr)

        # Frequency-wise correlation
        correlation = torch.corrcoef(torch.stack([
            original_magnitude.flatten(),
            reconstructed_magnitude.flatten()
        ]))[0, 1]
        spectral_metrics["spectral_correlation"] = float(correlation) if not torch.isnan(correlation) else 0.0

        # High frequency preservation (above 8kHz)
        sr = 44100  # Assume standard sample rate
        nyquist = sr // 2
        high_freq_bin = int(8000 * original_magnitude.shape[-1] / nyquist)

        if high_freq_bin < original_magnitude.shape[-1]:
            high_freq_original = original_magnitude[..., high_freq_bin:]
            high_freq_reconstructed = reconstructed_magnitude[..., high_freq_bin:]

            hf_signal_power = torch.mean(high_freq_original ** 2)
            hf_error_power = torch.mean((high_freq_original - high_freq_reconstructed) ** 2)
            hf_snr = 10 * torch.log10(hf_signal_power / (hf_error_power + 1e-10))
            spectral_metrics["high_freq_snr"] = float(hf_snr)

        return spectral_metrics

    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score (0-100) based on multiple metrics."""
        snr = metrics.get("reconstruction_snr", 0)
        sdr = metrics.get("source_distortion_ratio", 0)
        spectral_snr = metrics.get("spectral_snr", 0)
        spectral_corr = metrics.get("spectral_correlation", 0)

        # Weighted scoring
        snr_score = min(max(snr, 0) / 30.0, 1.0) * 40  # 40% weight on SNR
        sdr_score = min(max(sdr, 0) / 25.0, 1.0) * 30  # 30% weight on SDR
        spectral_score = min(max(spectral_snr, 0) / 25.0, 1.0) * 20  # 20% weight on spectral SNR
        corr_score = max(spectral_corr, 0) * 10  # 10% weight on correlation

        total_score = snr_score + sdr_score + spectral_score + corr_score
        return min(total_score, 100.0)

    def get_quality_assessment(self, metrics: Dict) -> str:
        """Get human-readable quality assessment."""
        snr = metrics.get("reconstruction_snr", 0)
        quality_score = metrics.get("overall_quality_score", 0)

        if quality_score >= 80:
            return "Excellent - Professional quality separation"
        elif quality_score >= 65:
            return "Very Good - High quality separation with minor artifacts"
        elif quality_score >= 50:
            return "Good - Acceptable quality for most applications"
        elif quality_score >= 35:
            return "Fair - Noticeable artifacts but usable"
        else:
            return "Poor - Significant artifacts present"