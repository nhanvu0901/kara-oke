import torch
import torchaudio
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# AudioCraft is optional - provide placeholder if not installed
try:
    from audiocraft.models import MusicGen, AudioGen
    from audiocraft.data.audio import audio_write

    AUDIOCRAFT_AVAILABLE = True
except ImportError:
    logger.warning("AudioCraft not installed. Using placeholder functionality.")
    AUDIOCRAFT_AVAILABLE = False


class AudioCraftProcessor:
    """Process audio using Meta's AudioCraft suite (or placeholder if not available)."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", "cpu")
        self.model = None

        if AUDIOCRAFT_AVAILABLE:
            self._load_model()
        else:
            logger.info("AudioCraft not available - using placeholder mode")

    def _load_model(self):
        """Load AudioCraft model if available."""
        if not AUDIOCRAFT_AVAILABLE:
            return

        try:
            model_name = self.config["audiocraft"]["model"]

            if "musicgen" in model_name:
                logger.info(f"Loading MusicGen model: {model_name}")
                if "musicgen" in model_name and not model_name.startswith("facebook/"):
                    model_name = f"facebook/{model_name}"
                self.model = MusicGen.get_pretrained(model_name, device=self.device)
            elif "audiogen" in model_name:
                logger.info(f"Loading AudioGen model: {model_name}")
                self.model = AudioGen.get_pretrained(model_name, device=self.device)
            else:
                raise ValueError(f"Unknown AudioCraft model: {model_name}")

            logger.info(f"AudioCraft model loaded on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load AudioCraft model: {e}")
            self.model = None

    def process(self,
                waveform: torch.Tensor,
                sample_rate: int,
                mode: str = "enhance",
                prompt: Optional[str] = None) -> Dict:
        """
        Process audio using AudioCraft or return placeholder.

        Args:
            waveform: Input audio
            sample_rate: Sample rate
            mode: Processing mode (enhance, generate, continue)
            prompt: Optional text prompt for generation

        Returns:
            Processed audio and metrics
        """

        # If AudioCraft is not available or model failed to load, return placeholder
        if not AUDIOCRAFT_AVAILABLE or not self.model:
            logger.info("Using placeholder AudioCraft processing (model not available)")

            # Apply simple processing as placeholder
            if mode == "enhance":
                # Simple enhancement: normalize audio
                output_audio = waveform / (torch.max(torch.abs(waveform)) + 1e-10)
            elif mode == "generate":
                # Generate silence as placeholder
                duration_samples = int(self.config["audiocraft"]["duration"] * sample_rate)
                output_audio = torch.zeros(2, duration_samples)
            else:  # continue mode
                # Just return original
                output_audio = waveform

            return {
                "audio": output_audio,
                "sample_rate": sample_rate,
                "mode": mode,
                "prompt": prompt,
                "metrics": {
                    "model": "placeholder",
                    "duration": float(output_audio.shape[-1] / sample_rate),
                    "audiocraft_available": False
                }
            }

        # Use actual AudioCraft if available
        try:
            if mode == "generate":
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
                if waveform.dim() == 2:
                    waveform = waveform.unsqueeze(0)
                elif waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0).unsqueeze(0)

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
                    "duration": float(output_audio.shape[-1] / output_sample_rate),
                    "audiocraft_available": True
                }
            }

        except Exception as e:
            logger.error(f"AudioCraft processing failed: {e}")
            logger.warning("Returning original audio as fallback")

            return {
                "audio": waveform,
                "sample_rate": sample_rate,
                "mode": mode,
                "prompt": prompt,
                "metrics": {
                    "model": self.config["audiocraft"]["model"],
                    "error": str(e),
                    "audiocraft_available": True
                }
            }