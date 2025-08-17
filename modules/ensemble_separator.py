"""
Enhanced Ensemble Source Separator
==================================
Production-ready ensemble separator combining multiple state-of-the-art models
for high-quality audio source separation with SNR > 0.20 target.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple, Union
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Import separation models
try:
    from demucs import pretrained
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    DEMUCS_AVAILABLE = True
except ImportError:
    logger.error("Demucs not installed. Install with: pip install demucs")
    DEMUCS_AVAILABLE = False

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("Librosa not installed. Some features will be limited.")
    LIBROSA_AVAILABLE = False


class SeparationModel(Enum):
    """Available separation models."""
    HTDEMUCS_6S = "htdemucs_6s"  # 6-stem high quality
    HTDEMUCS_FT = "htdemucs_ft"  # Fine-tuned 4-stem
    HTDEMUCS = "htdemucs"  # Standard 4-stem
    MDX_EXTRA_Q = "mdx_extra_q"  # MDX high quality
    MDX23C = "mdx23c_musdb18"  # Latest MDX variant


@dataclass
class ModelConfig:
    """Configuration for a separation model."""
    name: str
    weight: float
    stems: int
    quality_score: float
    optimal_for: List[str]


class ModelRegistry:
    """Registry of available models with their configurations."""

    MODELS = {
        SeparationModel.HTDEMUCS_6S: ModelConfig(
            name="htdemucs_6s",
            weight=1.0,
            stems=6,
            quality_score=0.95,
            optimal_for=["vocals", "drums", "bass", "piano", "guitar", "other"]
        ),
        SeparationModel.HTDEMUCS_FT: ModelConfig(
            name="htdemucs_ft",
            weight=0.85,
            stems=4,
            quality_score=0.90,
            optimal_for=["vocals", "drums", "bass", "other"]
        ),
        SeparationModel.HTDEMUCS: ModelConfig(
            name="htdemucs",
            weight=0.70,
            stems=4,
            quality_score=0.80,
            optimal_for=["vocals", "drums", "bass", "other"]
        ),
        SeparationModel.MDX_EXTRA_Q: ModelConfig(
            name="mdx_extra_q",
            weight=0.90,
            stems=4,
            quality_score=0.88,
            optimal_for=["vocals", "instrumental"]
        ),
        SeparationModel.MDX23C: ModelConfig(
            name="mdx23c_musdb18",
            weight=0.85,
            stems=4,
            quality_score=0.87,
            optimal_for=["vocals", "drums", "bass", "other"]
        )
    }


class EnsembleSourceSeparator:
    """
    Production-ready ensemble source separator combining multiple models
    for optimal separation quality (target SNR > 0.20).
    """

    def __init__(self, config: Dict):
        """
        Initialize the ensemble separator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = self._setup_device(config.get("device", "auto"))
        self.models = {}
        self.model_configs = {}

        # Performance settings
        self.timeout = config.get("timeout", 1800)
        self.max_parallel_models = config.get("max_parallel_models", 2)
        self.chunk_size = config.get("chunk_size", None)  # For memory-limited processing

        # Quality settings
        self.quality_mode = config.get("quality_mode", "highest")
        self.target_snr = config.get("target_snr", 0.20)

        # Ensemble configuration
        self.ensemble_models = self._get_ensemble_models()
        self.blend_method = config.get("blend_method", "weighted_average")
        self.use_frequency_weighting = config.get("use_frequency_weighting", True)

        # Cache for model outputs (memory optimization)
        self.enable_cache = config.get("enable_cache", True)
        self.model_cache = {} if self.enable_cache else None

        # Load models
        self._load_models()

    def _setup_device(self, device_config: str) -> str:
        """Setup the compute device."""
        if device_config == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using Apple Silicon MPS acceleration")
            elif torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("Using CPU (consider GPU for better performance)")
        else:
            device = device_config

        return device

    def _get_ensemble_models(self) -> List[str]:
        """Get list of models to use based on quality mode."""
        if "ensemble_models" in self.config:
            return self.config["ensemble_models"]

        if self.quality_mode == "highest":
            return ["htdemucs_6s", "htdemucs_ft", "mdx_extra_q"]
        elif self.quality_mode == "high":
            return ["htdemucs_ft", "mdx23c_musdb18"]
        elif self.quality_mode == "balanced":
            return ["htdemucs_ft"]
        else:  # fast
            return ["htdemucs"]

    def _load_models(self):
        """Load all models in the ensemble."""
        if not DEMUCS_AVAILABLE:
            raise RuntimeError("Demucs is required for source separation")

        logger.info(f"Loading ensemble models for {self.quality_mode} quality mode...")

        successful_loads = 0
        failed_loads = []

        for model_name in self.ensemble_models:
            try:
                # Get model config
                model_enum = None
                for enum_val in SeparationModel:
                    if enum_val.value == model_name:
                        model_enum = enum_val
                        break

                if not model_enum:
                    logger.warning(f"Unknown model: {model_name}")
                    continue

                model_config = ModelRegistry.MODELS[model_enum]

                logger.info(
                    f"Loading {model_name} (stems: {model_config.stems}, quality: {model_config.quality_score:.2f})...")

                # Load the model
                model = pretrained.get_model(model_name)

                # Move to device
                model.to(self.device)
                model.eval()

                # Store model and config
                self.models[model_name] = model
                self.model_configs[model_name] = model_config

                successful_loads += 1
                logger.info(f"✓ {model_name} loaded successfully")

            except Exception as e:
                logger.error(f"✗ Failed to load {model_name}: {str(e)}")
                failed_loads.append(model_name)

        if successful_loads == 0:
            raise RuntimeError("No models could be loaded")

        logger.info(f"Ensemble ready: {successful_loads} models loaded, {len(failed_loads)} failed")

        if failed_loads:
            logger.warning(f"Failed models: {', '.join(failed_loads)}")

    def separate(self,
                 waveform: torch.Tensor,
                 sample_rate: int,
                 progress_callback: Optional[Callable] = None) -> Dict:
        """
        Perform high-quality ensemble source separation.

        Args:
            waveform: Input audio tensor [channels, samples]
            sample_rate: Sample rate in Hz
            progress_callback: Optional callback(progress: float, message: str)

        Returns:
            Dictionary containing separated stems and metrics
        """
        start_time = time.time()

        try:
            # Validate input
            if waveform.dim() not in [1, 2]:
                raise ValueError(f"Expected 1D or 2D tensor, got {waveform.dim()}D")

            if progress_callback:
                progress_callback(0.05, "Preprocessing audio...")

            # Preprocess audio
            processed_audio, original_sr = self._preprocess_audio(waveform, sample_rate)

            # Clear cache if enabled
            if self.enable_cache:
                self.model_cache.clear()

            # Run ensemble models
            if progress_callback:
                progress_callback(0.10, f"Running {len(self.models)} models...")

            model_results = self._run_ensemble_models(processed_audio, progress_callback)

            if not model_results:
                raise RuntimeError("No models produced results")

            # Blend results
            if progress_callback:
                progress_callback(0.80, "Blending ensemble results...")

            final_stems = self._blend_ensemble_results(model_results, processed_audio)

            # Calculate metrics
            if progress_callback:
                progress_callback(0.90, "Calculating quality metrics...")

            metrics = self._calculate_metrics(processed_audio, final_stems)

            # Post-process stems
            if progress_callback:
                progress_callback(0.95, "Finalizing output...")

            final_stems = self._postprocess_stems(final_stems, original_sr, sample_rate)

            processing_time = time.time() - start_time

            # Prepare results
            results = {
                "stems": final_stems,
                "source_names": list(final_stems.keys()),
                "metrics": metrics,
                "processing_time": processing_time,
                "ensemble_info": {
                    "models_used": list(model_results.keys()),
                    "blend_method": self.blend_method,
                    "quality_mode": self.quality_mode,
                    "device": self.device
                }
            }

            if progress_callback:
                snr = metrics.get("reconstruction_snr", 0)
                progress_callback(1.0, f"Complete! SNR: {snr:.3f} dB")

            return results

        except Exception as e:
            logger.error(f"Separation failed: {str(e)}")
            raise

    def _preprocess_audio(self, waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, int]:
        """Preprocess audio for optimal separation."""
        original_sr = sample_rate

        # Ensure 2D tensor [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to stereo if mono
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            # Mix down to stereo if more than 2 channels
            waveform = waveform[:2, :]

        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.95

        # Resample to 44100 Hz for consistency across models
        if sample_rate != 44100:
            resampler = torchaudio.transforms.Resample(sample_rate, 44100)
            waveform = resampler(waveform)

        return waveform, original_sr

    def _run_ensemble_models(self, audio: torch.Tensor, progress_callback: Optional[Callable]) -> Dict:
        """Run all models in the ensemble."""
        model_results = {}
        total_models = len(self.models)

        if total_models == 0:
            raise RuntimeError("No models loaded")

        # Single model or sequential processing
        if total_models == 1 or self.max_parallel_models == 1:
            for i, (model_name, model) in enumerate(self.models.items()):
                if progress_callback:
                    progress = 0.10 + (0.70 * (i + 1) / total_models)
                    progress_callback(progress, f"Processing with {model_name}...")

                result = self._run_single_model(model, audio, model_name)
                if result:
                    model_results[model_name] = result

        else:
            # Parallel processing
            model_results = self._run_models_parallel(audio, progress_callback)

        return model_results

    def _run_models_parallel(self, audio: torch.Tensor, progress_callback: Optional[Callable]) -> Dict:
        """Run models in parallel for faster processing."""
        model_results = {}
        completed_count = 0
        total_models = len(self.models)
        progress_lock = threading.Lock()

        def update_progress():
            nonlocal completed_count
            with progress_lock:
                completed_count += 1
                if progress_callback:
                    progress = 0.10 + (0.70 * completed_count / total_models)
                    progress_callback(progress, f"Completed {completed_count}/{total_models} models")

        with ThreadPoolExecutor(max_workers=min(self.max_parallel_models, total_models)) as executor:
            future_to_model = {
                executor.submit(self._run_single_model, model, audio, model_name): model_name
                for model_name, model in self.models.items()
            }

            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result(timeout=self.timeout)
                    if result:
                        model_results[model_name] = result
                    update_progress()
                except Exception as e:
                    logger.error(f"Model {model_name} failed: {str(e)}")
                    update_progress()

        return model_results

    def _run_single_model(self, model, audio: torch.Tensor, model_name: str) -> Optional[Dict]:
        """Run a single separation model."""
        try:
            # Check cache first
            if self.enable_cache and model_name in self.model_cache:
                logger.info(f"Using cached result for {model_name}")
                return self.model_cache[model_name]

            # Prepare audio for model
            model_audio = convert_audio(
                audio.unsqueeze(0),  # Add batch dimension
                44100,  # Input sample rate
                model.samplerate,
                model.audio_channels
            )

            # Move to device
            model_audio = model_audio.to(self.device)

            # Apply model
            with torch.no_grad():
                sources = apply_model(
                    model,
                    model_audio,
                    device=self.device,
                    progress=False,
                    num_workers=0
                )

            # Extract stems
            stems = {}
            for i, name in enumerate(model.sources):
                stem_audio = sources[0, i].cpu()  # Remove batch dim and move to CPU

                # Resample back to 44100 if needed
                if model.samplerate != 44100:
                    resampler = torchaudio.transforms.Resample(model.samplerate, 44100)
                    stem_audio = resampler(stem_audio)

                stems[name] = stem_audio

            result = {
                "stems": stems,
                "source_names": model.sources,
                "model_config": self.model_configs[model_name]
            }

            # Cache result if enabled
            if self.enable_cache:
                self.model_cache[model_name] = result

            return result

        except Exception as e:
            logger.error(f"Error running {model_name}: {str(e)}")
            return None

    def _blend_ensemble_results(self, model_results: Dict, original_audio: torch.Tensor) -> Dict:
        """Blend results from multiple models using advanced techniques."""
        if len(model_results) == 1:
            # Single model, no blending needed
            return list(model_results.values())[0]["stems"]

        # Collect all unique stem names
        all_stems = set()
        for result in model_results.values():
            all_stems.update(result["stems"].keys())

        # Create stem mapping for normalization
        stem_mapping = self._create_stem_mapping(all_stems)

        blended_stems = {}

        for canonical_name, variations in stem_mapping.items():
            stem_data = self._collect_stem_data(canonical_name, variations, model_results)

            if stem_data:
                if self.blend_method == "weighted_average":
                    blended = self._weighted_average_blend(stem_data)
                elif self.blend_method == "median":
                    blended = self._median_blend(stem_data)
                elif self.blend_method == "frequency_weighted":
                    blended = self._frequency_weighted_blend(stem_data)
                else:
                    blended = self._weighted_average_blend(stem_data)

                blended_stems[canonical_name] = blended

        return blended_stems

    def _create_stem_mapping(self, stem_names: set) -> Dict[str, List[str]]:
        """Create mapping from canonical names to variations."""
        mapping = {
            "vocals": [],
            "drums": [],
            "bass": [],
            "other": [],
            "piano": [],
            "guitar": []
        }

        for stem in stem_names:
            stem_lower = stem.lower()
            if "vocal" in stem_lower or "voice" in stem_lower:
                mapping["vocals"].append(stem)
            elif "drum" in stem_lower:
                mapping["drums"].append(stem)
            elif "bass" in stem_lower:
                mapping["bass"].append(stem)
            elif "piano" in stem_lower:
                mapping["piano"].append(stem)
            elif "guitar" in stem_lower:
                mapping["guitar"].append(stem)
            else:
                mapping["other"].append(stem)

        # Remove empty mappings
        return {k: v for k, v in mapping.items() if v}

    def _collect_stem_data(self, canonical_name: str, variations: List[str],
                           model_results: Dict) -> List[Tuple[torch.Tensor, float]]:
        """Collect all versions of a stem with their weights."""
        stem_data = []

        for model_name, result in model_results.items():
            model_config = result["model_config"]

            for variation in variations:
                if variation in result["stems"]:
                    stem_tensor = result["stems"][variation]

                    # Calculate weight based on model quality and stem relevance
                    base_weight = model_config.weight

                    # Boost weight if this stem is in the model's optimal list
                    if canonical_name in model_config.optimal_for:
                        weight = base_weight * 1.2
                    else:
                        weight = base_weight

                    stem_data.append((stem_tensor, weight))
                    break

        return stem_data

    def _weighted_average_blend(self, stem_data: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """Blend stems using weighted average."""
        if not stem_data:
            return torch.zeros(2, 1)

        stems, weights = zip(*stem_data)
        weights = torch.tensor(weights)
        weights = weights / weights.sum()

        # Ensure all stems have same shape
        target_shape = stems[0].shape
        aligned_stems = []

        for stem in stems:
            if stem.shape != target_shape:
                # Align channels
                if stem.shape[0] != target_shape[0]:
                    if stem.shape[0] == 1 and target_shape[0] == 2:
                        stem = stem.repeat(2, 1)
                    elif stem.shape[0] == 2 and target_shape[0] == 1:
                        stem = stem.mean(dim=0, keepdim=True)

                # Align length
                if stem.shape[-1] != target_shape[-1]:
                    min_len = min(stem.shape[-1], target_shape[-1])
                    stem = stem[..., :min_len]

            aligned_stems.append(stem)

        # Weighted sum
        blended = torch.zeros_like(aligned_stems[0])
        for stem, weight in zip(aligned_stems, weights):
            blended += stem * weight

        return blended

    def _median_blend(self, stem_data: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """Blend stems using median (robust to outliers)."""
        if not stem_data:
            return torch.zeros(2, 1)

        stems, _ = zip(*stem_data)

        # Stack and take median
        stacked = torch.stack([s for s in stems if s.shape == stems[0].shape])
        if len(stacked) > 0:
            return torch.median(stacked, dim=0)[0]
        else:
            return stems[0]

    def _frequency_weighted_blend(self, stem_data: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """Blend stems with frequency-domain weighting."""
        if not stem_data or not self.use_frequency_weighting:
            return self._weighted_average_blend(stem_data)

        stems, weights = zip(*stem_data)
        weights = torch.tensor(weights)
        weights = weights / weights.sum()

        # Convert to frequency domain
        stfts = []
        for stem in stems:
            if stem.dim() > 1:
                mono_stem = stem.mean(dim=0)
            else:
                mono_stem = stem

            stft = torch.stft(
                mono_stem,
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                return_complex=True,
                window=torch.hann_window(2048)
            )
            stfts.append(stft)

        # Weighted blend in frequency domain
        blended_stft = torch.zeros_like(stfts[0], dtype=torch.complex64)
        for stft, weight in zip(stfts, weights):
            blended_stft += stft * weight

        # Convert back to time domain
        blended_mono = torch.istft(
            blended_stft,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window=torch.hann_window(2048),
            length=stems[0].shape[-1]
        )

        # Restore stereo if needed
        if stems[0].dim() > 1 and stems[0].shape[0] == 2:
            blended = blended_mono.unsqueeze(0).repeat(2, 1)
        else:
            blended = blended_mono

        return blended

    def _calculate_metrics(self, original: torch.Tensor, stems: Dict) -> Dict:
        """Calculate separation quality metrics."""
        metrics = {}

        # Reconstruct from stems
        reconstructed = torch.zeros_like(original)
        for stem in stems.values():
            if stem.shape == original.shape:
                reconstructed += stem

        # SNR calculation
        signal_power = torch.mean(original ** 2)
        noise = original - reconstructed
        noise_power = torch.mean(noise ** 2)

        if signal_power > 0:
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
            metrics["reconstruction_snr"] = float(snr)
        else:
            metrics["reconstruction_snr"] = 0.0

        # SDR (Signal to Distortion Ratio) - more accurate than SNR
        if signal_power > 0:
            sdr = 10 * torch.log10(torch.sum(original ** 2) / (torch.sum(noise ** 2) + 1e-10))
            metrics["sdr"] = float(sdr)

        # Per-stem metrics
        total_energy = torch.sum(original ** 2)
        for name, stem in stems.items():
            stem_energy = torch.sum(stem ** 2)
            metrics[f"{name}_energy"] = float(stem_energy)
            metrics[f"{name}_energy_ratio"] = float(stem_energy / (total_energy + 1e-10))

        # Quality assessment
        metrics["num_stems"] = len(stems)
        metrics["quality_grade"] = self._assess_quality(metrics["reconstruction_snr"])
        metrics["target_achieved"] = metrics["reconstruction_snr"] >= self.target_snr

        return metrics

    def _assess_quality(self, snr: float) -> str:
        """Assess separation quality based on SNR."""
        if snr >= 0.30:
            return "Excellent"
        elif snr >= 0.20:
            return "Good"
        elif snr >= 0.10:
            return "Fair"
        else:
            return "Poor"

    def _postprocess_stems(self, stems: Dict, original_sr: int, target_sr: int) -> Dict:
        """Post-process stems for final output."""
        processed_stems = {}

        for name, stem in stems.items():
            # Resample to target sample rate if needed
            if original_sr != target_sr and target_sr != 44100:
                resampler = torchaudio.transforms.Resample(44100, target_sr)
                stem = resampler(stem)

            # Normalize to prevent clipping
            max_val = torch.max(torch.abs(stem))
            if max_val > 0.99:
                stem = stem * (0.99 / max_val)

            # Remove DC offset
            stem = stem - torch.mean(stem, dim=-1, keepdim=True)

            processed_stems[name] = stem

        return processed_stems

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        info = {
            "loaded_models": list(self.models.keys()),
            "model_configs": {
                name: {
                    "weight": config.weight,
                    "stems": config.stems,
                    "quality_score": config.quality_score
                }
                for name, config in self.model_configs.items()
            },
            "device": self.device,
            "quality_mode": self.quality_mode,
            "target_snr": self.target_snr,
            "blend_method": self.blend_method
        }
        return info