import torch
import torchaudio
from typing import Dict, Optional
import logging

# Import AudioCraft components (requires: pip install audiocraft)


logger = logging.getLogger(__name__)
try:
    from audiocraft.models import MusicGen, AudioGen
    from audiocraft.data.audio import audio_write

    AUDIOCRAFT_AVAILABLE = True
except ImportError:
    logger.warning("AudioCraft not installed. Install with: pip install audiocraft")
    AUDIOCRAFT_AVAILABLE = False

class AudioCraftProcessor:
    """Process audio using Meta's AudioCraft suite."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", "cpu")
        self.model = None

        if AUDIOCRAFT_AVAILABLE:
            self._load_model()

    def _load_model(self):
        """Load AudioCraft model."""
        try:
            model_name = self.config["audiocraft"]["model"]

            if "musicgen" in model_name:
                logger.info(f"Loading MusicGen model: {model_name}")
                if "musicgen" in model_name and not model_name.startswith("facebook/"):
                    model_name = f"facebook/{model_name}"
                logger.info(f"Loading MusicGen model: {model_name}")
                self.model = MusicGen.get_pretrained(model_name)
            elif "audiogen" in model_name:
                logger.info(f"Loading AudioGen model: {model_name}")
                self.model = AudioGen.get_pretrained(model_name)
            else:
                raise ValueError(f"Unknown AudioCraft model: {model_name}")

            self.model.to(self.device)
            logger.info(f"AudioCraft model loaded on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load AudioCraft model: {e}")
            raise

    def process(self,
                waveform: torch.Tensor,
                sample_rate: int,
                mode: str = "enhance",
                prompt: Optional[str] = None) -> Dict:
        """
        Process audio using AudioCraft.

        Args:
            waveform: Input audio
            sample_rate: Sample rate
            mode: Processing mode (enhance, generate, continue)
            prompt: Optional text prompt for generation

        Returns:
            Processed audio and metrics
        """
        if not self.model:
            raise RuntimeError("AudioCraft model not loaded")

        try:
            if mode == "generate":
                # Generate new audio from text prompt
                if not prompt:
                    prompt = "upbeat electronic music with synth leads"

                self.model.set_generation_params(
                    duration=self.config["audiocraft"]["duration"],
                    temperature=self.config["audiocraft"]["temperature"],
                    top_k=self.config["audiocraft"]["top_k"],
                    top_p=self.config["audiocraft"]["top_p"],
                )

                with torch.no_grad():
                    generated = self.model.generate(
                        descriptions=[prompt],
                        progress=True
                    )

                output_audio = generated[0]

            elif mode == "continue":
                # Continue/extend existing audio
                # Prepare audio for model
                if waveform.dim() == 2:
                    waveform = waveform.unsqueeze(0)  # Add batch dimension

                with torch.no_grad():
                    continued = self.model.generate_continuation(
                        waveform,
                        prompt_sample_rate=sample_rate,
                        progress=True
                    )

                output_audio = continued[0]

            else:  # enhance mode
                # Process/enhance existing audio
                # This is a simplified example - actual enhancement would be more complex
                output_audio = waveform

            return {
                "audio": output_audio,
                "sample_rate": self.model.sample_rate if hasattr(self.model, 'sample_rate') else sample_rate,
                "mode": mode,
                "prompt": prompt,
                "metrics": {
                    "model": self.config["audiocraft"]["model"],
                    "duration": float(output_audio.shape[-1] / sample_rate)
                }
            }

        except Exception as e:
            logger.error(f"AudioCraft processing failed: {e}")
            raise