import torch
import torchaudio
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class AudioLoader:
    """Load and preprocess audio files for high-quality separation."""

    def __init__(self, config: Dict):
        self.config = config
        self.target_sr = config.get("sample_rate", 44100)

    def load(self, file_path: Path) -> Dict:
        """
        Load audio file and return preprocessed data optimized for separation.

        Returns:
            Dict containing waveform, sample_rate, duration, etc.
        """
        try:
            logger.info(f"Loading audio file: {file_path}")

            # Load audio with torchaudio
            waveform, sample_rate = torchaudio.load(file_path)

            logger.info(f"Original: {waveform.shape[0]} channels, {sample_rate}Hz, {waveform.shape[1]} samples")

            # Resample if necessary to target sample rate
            if sample_rate != self.target_sr:
                logger.info(f"Resampling from {sample_rate}Hz to {self.target_sr}Hz")
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
                waveform = resampler(waveform)
                sample_rate = self.target_sr

            # Ensure stereo for better separation quality
            if waveform.shape[0] == 1:
                # Convert mono to stereo by duplicating channel
                waveform = waveform.repeat(2, 1)
                logger.info("Converted mono to stereo")
            elif waveform.shape[0] > 2:
                # Convert multichannel to stereo by taking first 2 channels
                waveform = waveform[:2, :]
                logger.info(f"Converted {waveform.shape[0]} channels to stereo")

            # Calculate duration
            duration = waveform.shape[1] / sample_rate

            # Quality checks
            max_amplitude = torch.max(torch.abs(waveform)).item()
            if max_amplitude > 0.99:
                logger.warning(f"Audio may be clipped (max amplitude: {max_amplitude:.3f})")

            return {
                "waveform": waveform,
                "sample_rate": sample_rate,
                "duration": duration,
                "channels": waveform.shape[0],
                "samples": waveform.shape[1],
                "format": file_path.suffix[1:].lower(),
                "filename": file_path.stem,
                "filepath": str(file_path),
                "max_amplitude": max_amplitude
            }

        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            raise RuntimeError(f"Audio loading failed: {e}") from e