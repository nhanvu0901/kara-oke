import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

# Import required libraries with fallbacks
try:
    from demucs import pretrained
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    DEMUCS_AVAILABLE = True
except ImportError:
    logger.warning("Demucs not available. Install with: pip install demucs")
    DEMUCS_AVAILABLE = False

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("Librosa not available. Install with: pip install librosa")
    LIBROSA_AVAILABLE = False


class EnsembleSourceSeparator:
    """
    High-quality ensemble source separator combining multiple models.
    Focuses on achieving SNR > 0.20 through model ensemble and advanced blending.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", "mps" if torch.backends.mps.is_available() else "cpu")
        self.models = {}
        self.model_weights = {}

        # Extended timeout for quality processing
        self.timeout = config.get("timeout", 1800)  # 30 minutes default

        # Quality settings
        self.quality_mode = config.get("quality_mode", "highest")
        self.target_snr = config.get("target_snr", 0.20)

        # Ensemble configuration
        self.ensemble_models = config.get("ensemble_models", [
            "htdemucs_6s",  # Best overall quality
            "htdemucs_ft",  # Fine-tuned version
            "mdx_extra_q",  # High-quality MDX
            "mdx23c_musdb18"  # Latest MDX variant
        ])

        self.blend_method = config.get("blend_method", "weighted_average")
        self.use_frequency_weighting = config.get("use_frequency_weighting", True)

        # Load models
        self._load_models()

    def _load_models(self):
        """Load all available models for ensemble processing."""
        logger.info("Loading ensemble models for high-quality separation...")

        if not DEMUCS_AVAILABLE:
            logger.error("Demucs is required for ensemble separation")
            raise RuntimeError("Demucs not available")

        # Available Demucs models with quality ratings
        available_models = {
            "htdemucs_6s": {"weight": 1.0, "stems": 6},  # Highest quality
            "htdemucs_ft": {"weight": 0.8, "stems": 4},  # Fine-tuned
            "htdemucs": {"weight": 0.6, "stems": 4},  # Standard
            "mdx_extra_q": {"weight": 0.9, "stems": 4},  # MDX high quality
            "mdx23c_musdb18": {"weight": 0.85, "stems": 4}  # Latest MDX
        }

        for model_name in self.ensemble_models:
            if model_name in available_models:
                try:
                    logger.info(f"Loading {model_name}...")
                    model = pretrained.get_model(model_name)
                    model.to(self.device)
                    model.eval()

                    self.models[model_name] = model
                    self.model_weights[model_name] = available_models[model_name]["weight"]

                    logger.info(f"✓ {model_name} loaded (weight: {available_models[model_name]['weight']})")

                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue

        if not self.models:
            raise RuntimeError("No models could be loaded for ensemble separation")

        logger.info(f"Ensemble ready with {len(self.models)} models")

    def separate(self,
                 waveform: torch.Tensor,
                 sample_rate: int,
                 progress_callback: Optional[Callable] = None) -> Dict:
        """
        Perform high-quality ensemble source separation.

        Args:
            waveform: Input audio tensor
            sample_rate: Sample rate
            progress_callback: Progress callback function

        Returns:
            Dictionary with separated stems and quality metrics
        """
        start_time = time.time()

        if progress_callback:
            progress_callback(0.05, "Initializing ensemble separation...")

        # Prepare audio for processing
        processed_audio = self._preprocess_audio(waveform, sample_rate)

        # Run models in parallel for speed
        model_results = self._run_ensemble_models(processed_audio, progress_callback)

        if progress_callback:
            progress_callback(0.80, "Blending ensemble results...")

        # Blend results using advanced techniques
        final_stems = self._blend_ensemble_results(model_results, processed_audio)

        if progress_callback:
            progress_callback(0.95, "Calculating quality metrics...")

        # Calculate comprehensive metrics
        metrics = self._calculate_advanced_metrics(processed_audio, final_stems)

        # Post-process stems for optimal quality
        final_stems = self._postprocess_stems(final_stems, sample_rate)

        processing_time = time.time() - start_time

        if progress_callback:
            progress_callback(1.0, f"Complete! SNR: {metrics.get('reconstruction_snr', 0):.3f}")

        return {
            "stems": final_stems,
            "source_names": list(final_stems.keys()),
            "metrics": metrics,
            "processing_time": processing_time,
            "ensemble_info": {
                "models_used": list(self.models.keys()),
                "blend_method": self.blend_method,
                "quality_mode": self.quality_mode
            }
        }

    def _preprocess_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Preprocess audio for optimal separation quality."""
        # Ensure stereo
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).repeat(2, 1)
        elif waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # Normalize to prevent clipping while preserving dynamics
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0.95:
            waveform = waveform * (0.95 / max_val)

        # Apply gentle high-pass filter to remove subsonic content
        if LIBROSA_AVAILABLE and sample_rate != 44100:
            # Resample to standard rate for consistency
            resampler = torchaudio.transforms.Resample(sample_rate, 44100)
            waveform = resampler(waveform)

        return waveform

    def _run_ensemble_models(self, audio: torch.Tensor, progress_callback: Optional[Callable]) -> Dict:
        """Run all models in the ensemble."""
        model_results = {}
        total_models = len(self.models)

        # Use ThreadPoolExecutor for parallel processing if multiple models
        if len(self.models) > 1:
            model_results = self._run_models_parallel(audio, progress_callback)
        else:
            # Single model execution
            for i, (model_name, model) in enumerate(self.models.items()):
                if progress_callback:
                    progress = 0.1 + (0.7 * i / total_models)
                    progress_callback(progress, f"Processing with {model_name}...")

                result = self._run_single_model(model, audio, model_name)
                if result:
                    model_results[model_name] = result

        return model_results

    def _run_models_parallel(self, audio: torch.Tensor, progress_callback: Optional[Callable]) -> Dict:
        """Run models in parallel for faster processing."""
        model_results = {}
        completed_count = 0
        total_models = len(self.models)

        # Create a lock for thread-safe progress updates
        progress_lock = threading.Lock()

        def update_progress():
            nonlocal completed_count
            with progress_lock:
                completed_count += 1
                if progress_callback:
                    progress = 0.1 + (0.7 * completed_count / total_models)
                    progress_callback(progress, f"Completed {completed_count}/{total_models} models")

        with ThreadPoolExecutor(max_workers=min(len(self.models), 3)) as executor:
            # Submit all model tasks
            future_to_model = {
                executor.submit(self._run_single_model, model, audio, model_name): model_name
                for model_name, model in self.models.items()
            }

            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result(timeout=self.timeout)
                    if result:
                        model_results[model_name] = result
                    update_progress()
                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {e}")
                    update_progress()

        return model_results

    def _run_single_model(self, model, audio: torch.Tensor, model_name: str) -> Optional[Dict]:
        """Run a single model and return results."""
        try:
            # Convert audio for the specific model
            model_audio = convert_audio(
                audio.unsqueeze(0),  # Add batch dimension
                44100,  # Assume preprocessed to 44100
                model.samplerate,
                model.audio_channels
            )

            # Apply model with no gradient computation
            with torch.no_grad():
                sources = apply_model(
                    model,
                    model_audio,
                    device=self.device,
                    progress=False,
                    num_workers=0  # Avoid conflicts in parallel execution
                )

            # Extract stems
            stems = {}
            source_names = model.sources

            for i, name in enumerate(source_names):
                stem_audio = sources[0, i]  # Remove batch dimension

                # Convert back to original sample rate if needed
                if model.samplerate != 44100:
                    resampler = torchaudio.transforms.Resample(model.samplerate, 44100)
                    stem_audio = resampler(stem_audio)

                stems[name] = stem_audio

            return {
                "stems": stems,
                "source_names": source_names,
                "model_name": model_name,
                "weight": self.model_weights[model_name]
            }

        except Exception as e:
            logger.error(f"Model {model_name} processing failed: {e}")
            return None

    def _blend_ensemble_results(self, model_results: Dict, original_audio: torch.Tensor) -> Dict:
        """Blend results from multiple models using advanced techniques."""
        if not model_results:
            raise RuntimeError("No model results to blend")

        if len(model_results) == 1:
            # Single model result
            return list(model_results.values())[0]["stems"]

        # Collect all stem names
        all_stem_names = set()
        for result in model_results.values():
            all_stem_names.update(result["stems"].keys())

        # Normalize stem names (handle variations)
        stem_mapping = self._create_stem_mapping(all_stem_names)

        blended_stems = {}

        for canonical_name in stem_mapping.keys():
            blended_stems[canonical_name] = self._blend_single_stem(
                canonical_name, stem_mapping[canonical_name], model_results, original_audio
            )

        return blended_stems

    def _create_stem_mapping(self, stem_names: set) -> Dict:
        """Create mapping from canonical stem names to variations."""
        canonical_mapping = {
            "vocals": ["vocals", "vocal", "voice"],
            "drums": ["drums", "drum"],
            "bass": ["bass"],
            "other": ["other", "accompaniment", "rest"],
            "piano": ["piano"],
            "guitar": ["guitar"]
        }

        stem_mapping = {}

        for canonical, variations in canonical_mapping.items():
            mapped_names = []
            for stem_name in stem_names:
                if stem_name.lower() in variations:
                    mapped_names.append(stem_name)

            if mapped_names:
                stem_mapping[canonical] = mapped_names

        return stem_mapping

    def _blend_single_stem(self, canonical_name: str, stem_variations: List[str],
                           model_results: Dict, original_audio: torch.Tensor) -> torch.Tensor:
        """Blend a single stem type across all models."""
        stem_tensors = []
        weights = []

        # Collect all versions of this stem
        for model_name, result in model_results.items():
            model_weight = result["weight"]

            for variation in stem_variations:
                if variation in result["stems"]:
                    stem_tensor = result["stems"][variation]

                    # Quality-based weighting
                    quality_weight = self._calculate_stem_quality_weight(
                        stem_tensor, original_audio, canonical_name
                    )

                    final_weight = model_weight * quality_weight

                    stem_tensors.append(stem_tensor)
                    weights.append(final_weight)
                    break

        if not stem_tensors:
            # Return silence if no stems found
            return torch.zeros_like(original_audio)

        # Normalize weights
        weights = torch.tensor(weights)
        weights = weights / torch.sum(weights)

        # Weighted average in frequency domain for better quality
        if self.use_frequency_weighting and len(stem_tensors) > 1:
            return self._frequency_domain_blend(stem_tensors, weights)
        else:
            # Simple weighted average
            blended = torch.zeros_like(stem_tensors[0])
            for stem, weight in zip(stem_tensors, weights):
                blended += stem * weight
            return blended

    def _calculate_stem_quality_weight(self, stem: torch.Tensor,
                                       original: torch.Tensor, stem_type: str) -> float:
        """Calculate quality-based weight for a stem."""
        try:
            # Energy-based quality assessment
            stem_energy = torch.mean(stem ** 2)

            # Frequency distribution quality (simplified)
            if LIBROSA_AVAILABLE:
                stem_np = stem.mean(dim=0).numpy() if stem.dim() > 1 else stem.numpy()

                # Spectral centroid as quality indicator
                if len(stem_np) > 1024:  # Minimum length for analysis
                    spectral_centroid = librosa.feature.spectral_centroid(y=stem_np, sr=44100)
                    centroid_mean = np.mean(spectral_centroid)

                    # Stem-specific quality adjustments
                    if stem_type == "vocals" and 1000 <= centroid_mean <= 4000:
                        return 1.2  # Boost good vocal range
                    elif stem_type == "drums" and stem_energy > 0.01:
                        return 1.1  # Boost energetic drums
                    elif stem_type == "bass" and centroid_mean < 500:
                        return 1.1  # Boost low-frequency bass

            # Energy threshold
            if stem_energy > 0.001:
                return 1.0
            else:
                return 0.5  # Reduce weight for very quiet stems

        except Exception:
            return 1.0

    def _frequency_domain_blend(self, stems: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        """Blend stems in frequency domain for better quality."""
        # Convert to frequency domain
        stfts = []
        for stem in stems:
            if stem.dim() > 1:
                # Average channels for processing
                mono_stem = torch.mean(stem, dim=0)
            else:
                mono_stem = stem

            stft = torch.stft(mono_stem, n_fft=2048, hop_length=512,
                              win_length=2048, return_complex=True)
            stfts.append(stft)

        # Weighted blend in frequency domain
        blended_stft = torch.zeros_like(stfts[0], dtype=torch.complex64)
        for stft, weight in zip(stfts, weights):
            blended_stft += stft * weight

        # Convert back to time domain
        blended_mono = torch.istft(blended_stft, n_fft=2048, hop_length=512,
                                   win_length=2048, length=stems[0].shape[-1])

        # Restore original channel structure
        if stems[0].dim() > 1:
            blended = blended_mono.unsqueeze(0).repeat(stems[0].shape[0], 1)
        else:
            blended = blended_mono

        return blended

    def _postprocess_stems(self, stems: Dict, sample_rate: int) -> Dict:
        """Post-process stems for optimal quality."""
        processed_stems = {}

        for stem_name, stem_audio in stems.items():
            # Gentle limiting to prevent clipping
            max_val = torch.max(torch.abs(stem_audio))
            if max_val > 0.99:
                stem_audio = stem_audio * (0.99 / max_val)

            # DC offset removal
            stem_audio = stem_audio - torch.mean(stem_audio)

            processed_stems[stem_name] = stem_audio

        return processed_stems

    def _calculate_advanced_metrics(self, original: torch.Tensor, stems: Dict) -> Dict:
        """Calculate comprehensive separation quality metrics."""
        metrics = {}

        # Reconstruct from stems
        reconstructed = torch.zeros_like(original)
        for stem in stems.values():
            if stem.shape != original.shape:
                # Handle shape mismatch
                if stem.shape[0] != original.shape[0]:
                    if stem.shape[0] == 1 and original.shape[0] == 2:
                        stem = stem.repeat(2, 1)
                    elif stem.shape[0] == 2 and original.shape[0] == 1:
                        stem = torch.mean(stem, dim=0, keepdim=True)

                # Handle length mismatch
                if stem.shape[-1] != original.shape[-1]:
                    min_len = min(stem.shape[-1], original.shape[-1])
                    stem = stem[..., :min_len]

            reconstructed += stem

        # SNR calculation
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean((original - reconstructed) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))

        metrics["reconstruction_snr"] = float(snr)
        metrics["num_stems"] = len(stems)

        # Per-stem energy analysis
        total_energy = torch.mean(original ** 2)
        for name, stem in stems.items():
            stem_energy = torch.mean(stem ** 2)
            metrics[f"{name}_energy"] = float(stem_energy)
            metrics[f"{name}_energy_ratio"] = float(stem_energy / (total_energy + 1e-10))

        # Quality assessment
        metrics["quality_grade"] = self._assess_quality_grade(snr, stems)
        metrics["target_achieved"] = float(snr) >= self.target_snr

        return metrics

    def _assess_quality_grade(self, snr: torch.Tensor, stems: Dict) -> str:
        """Assess overall separation quality grade."""
        snr_val = float(snr)

        if snr_val >= 0.30:
            return "Excellent"
        elif snr_val >= 0.20:
            return "Good"
        elif snr_val >= 0.10:
            return "Fair"
        else:
            return "Poor"

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            "loaded_models": list(self.models.keys()),
            "model_weights": self.model_weights,
            "device": self.device,
            "quality_mode": self.quality_mode,
            "target_snr": self.target_snr
        }