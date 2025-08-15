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
                # MusicGen handles device placement internally
                # The device parameter can be passed during model loading
                self.model = MusicGen.get_pretrained(model_name, device=self.device)
            elif "audiogen" in model_name:
                logger.info(f"Loading AudioGen model: {model_name}")
                self.model = AudioGen.get_pretrained(model_name, device=self.device)
            else:
                raise ValueError(f"Unknown AudioCraft model: {model_name}")


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
                elif waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

                # Ensure waveform is on the correct device
                if hasattr(self.model, 'device'):
                    waveform = waveform.to(self.model.device)

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
                # For now, just return the original audio
                # In a real implementation, you might use AudioGen or other processing
                output_audio = waveform
                logger.info("Enhancement mode is a placeholder - returning original audio")

            # Get the sample rate from the model if available
            if hasattr(self.model, 'sample_rate'):
                output_sample_rate = self.model.sample_rate
            elif hasattr(self.model, 'compression_model') and hasattr(self.model.compression_model, 'sample_rate'):
                output_sample_rate = self.model.compression_model.sample_rate
            else:
                output_sample_rate = sample_rate

            return {
                "audio": output_audio,
                "sample_rate": output_sample_rate,
                "mode": mode,
                "prompt": prompt,
                "metrics": {
                    "model": self.config["audiocraft"]["model"],
                    "duration": float(output_audio.shape[-1] / output_sample_rate)
                }
            }

        except Exception as e:
            logger.error(f"AudioCraft processing failed: {e}")
            # Return original audio as fallback
            logger.warning("Returning original audio as fallback")
            return {
                "audio": waveform,
                "sample_rate": sample_rate,
                "mode": mode,
                "prompt": prompt,
                "metrics": {
                    "model": self.config["audiocraft"]["model"],
                    "error": str(e)
                }
            }