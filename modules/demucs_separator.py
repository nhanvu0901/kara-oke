import torch
import torchaudio
from pathlib import Path
from typing import Dict, Optional, Callable
import logging
import time

# Import Demucs (requires: pip install demucs)


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
        Separate audio into stems using Demucs v4.

        Args:
            waveform: Input audio tensor
            sample_rate: Sample rate
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with separated stems and metrics
        """
        if not self.model:
            raise RuntimeError("Demucs model not loaded")

        start_time = time.time()

        try:
            # Prepare audio for Demucs
            audio = convert_audio(
                waveform.unsqueeze(0),  # Add batch dimension
                sample_rate,
                self.model.samplerate,
                self.model.audio_channels
            )

            # Apply model
            with torch.no_grad():
                sources = apply_model(
                    self.model,
                    audio,
                    device=self.device,
                    progress=True if progress_callback else False
                )

            # Extract stems
            stems = {}
            source_names = self.model.sources

            for i, name in enumerate(source_names):
                stem_audio = sources[0, i]  # Remove batch dimension

                # Convert back to original sample rate if needed
                if self.model.samplerate != sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        self.model.samplerate,
                        sample_rate
                    )
                    stem_audio = resampler(stem_audio)

                stems[name] = stem_audio

            # Calculate separation metrics (educational)
            metrics = self._calculate_metrics(waveform, stems)

            processing_time = time.time() - start_time

            return {
                "stems": stems,
                "source_names": source_names,
                "metrics": metrics,
                "processing_time": processing_time,
                "model": self.model_name
            }

        except Exception as e:
            logger.error(f"Separation failed: {e}")
            raise

    def _calculate_metrics(self, original: torch.Tensor, stems: Dict) -> Dict:
        """Calculate educational metrics for source separation quality."""
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

        # Calculate SNR
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean((original - reconstructed) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))

        metrics["reconstruction_snr"] = float(snr)
        metrics["num_stems"] = len(stems)

        # Energy distribution across stems
        for name, stem in stems.items():
            energy = torch.mean(stem ** 2)
            metrics[f"{name}_energy"] = float(energy)

        return metrics