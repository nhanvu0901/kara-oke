import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AudioLoader:
    """Load and preprocess audio files for the pipeline."""

    def __init__(self, config: Dict):
        self.config = config
        self.target_sr = config.get("sample_rate", 44100)

    def load(self, file_path: Path) -> Dict:
        """
        Load audio file and return preprocessed data.

        Returns:
            Dict containing waveform, sample_rate, duration, etc.
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(file_path)

            # Resample if necessary
            if sample_rate != self.target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
                waveform = resampler(waveform)
                sample_rate = self.target_sr

            # Convert to mono if needed for certain processing
            mono_waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Calculate duration
            duration = waveform.shape[1] / sample_rate

            return {
                "waveform": waveform,
                "mono_waveform": mono_waveform,
                "sample_rate": sample_rate,
                "duration": duration,
                "channels": waveform.shape[0],
                "format": file_path.suffix[1:],
                "filename": file_path.stem,
                "filepath": str(file_path)
            }

        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            raise
