import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable
import logging
import time

import torchaudio

logger = logging.getLogger(__name__)


class UVR5Separator:
    """UVR5 MDX-Net separator for high-quality instrumental separation."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", "cpu")
        self.model_name = config["uvr5"]["model"]
        self.chunk_size = config["uvr5"].get("chunk_size", 261120)
        self.overlap = config["uvr5"].get("overlap", 0.5)
        # Model loading logic here

    def separate(self, waveform: torch.Tensor, sample_rate: int,
                 progress_callback: Optional[Callable] = None) -> Dict:
        """
        Separate audio using UVR5 MDX-Net architecture.
        Optimized for instrumental-only separation (no vocals).
        """
        start_time = time.time()

        if not self.model:
            raise RuntimeError("UVR5 model not loaded")

        try:
            # Progress tracking
            if progress_callback:
                progress_callback(0.05, "Preparing audio for UVR5...")

            # Convert to numpy and ensure stereo
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)

            audio_np = waveform.numpy()

            # Resample if needed (MDX-Net expects 44100)
            if sample_rate != 44100:
                if progress_callback:
                    progress_callback(0.1, "Resampling audio...")
                resampler = torchaudio.transforms.Resample(sample_rate, 44100)
                waveform_resampled = resampler(waveform)
                audio_np = waveform_resampled.numpy()
                process_sr = 44100
            else:
                process_sr = sample_rate

            # Normalize audio
            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / max_val

            # Pad audio for processing
            if progress_callback:
                progress_callback(0.15, "Padding audio...")

            pad_size = self.margin
            audio_padded = np.pad(audio_np, ((0, 0), (pad_size, pad_size)), mode='reflect')

            # Prepare for chunk processing
            chunk_size = self.chunk_size
            overlap_size = int(chunk_size * self.overlap)
            step_size = chunk_size - overlap_size

            # Calculate number of chunks
            total_length = audio_padded.shape[1]
            num_chunks = max(1, int(np.ceil((total_length - chunk_size) / step_size)) + 1)

            # Initialize output arrays for stems
            stems_dict = {
                'drums': np.zeros_like(audio_padded),
                'bass': np.zeros_like(audio_padded),
                'other': np.zeros_like(audio_padded)  # Keys, guitars, etc.
            }

            # Process chunks
            for i in range(num_chunks):
                if progress_callback:
                    progress = 0.15 + (i / num_chunks) * 0.7
                    progress_callback(progress, f"Processing chunk {i + 1}/{num_chunks}...")

                # Extract chunk
                start_idx = i * step_size
                end_idx = min(start_idx + chunk_size, total_length)
                chunk = audio_padded[:, start_idx:end_idx]

                # Pad chunk if it's smaller than chunk_size
                if chunk.shape[1] < chunk_size:
                    chunk = np.pad(chunk, ((0, 0), (0, chunk_size - chunk.shape[1])), mode='constant')

                # Convert to frequency domain
                chunk_tensor = torch.from_numpy(chunk).float()

                # STFT processing
                n_fft = 6144
                hop_length = 1024
                window = torch.hann_window(n_fft).to(chunk_tensor.device)

                # Compute STFT
                stft = torch.stft(
                    chunk_tensor,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window,
                    center=True,
                    return_complex=True
                )

                # Convert to magnitude and phase
                magnitude = torch.abs(stft)
                phase = torch.angle(stft)

                # Prepare input for ONNX model (MDX-Net expects specific shape)
                mag_numpy = magnitude.numpy()

                # Reshape for model input [batch, channels, freq_bins, time_frames]
                if mag_numpy.ndim == 3:
                    mag_input = mag_numpy[np.newaxis, :, :, :]
                else:
                    mag_input = mag_numpy

                # Run inference for each stem type
                for stem_name in stems_dict.keys():
                    if progress_callback:
                        sub_progress = 0.15 + (i / num_chunks) * 0.7 + (0.7 / (num_chunks * 3))
                        progress_callback(sub_progress, f"Extracting {stem_name} from chunk {i + 1}...")

                    # Get appropriate model for stem
                    model = self._get_model_for_stem(stem_name)

                    if model is not None:
                        # Run ONNX inference
                        input_name = model.get_inputs()[0].name
                        output_name = model.get_outputs()[0].name

                        # Inference with TTA (Test Time Augmentation) if enabled
                        if self.config["uvr5"].get("tta", False):
                            # Original
                            pred1 = model.run([output_name], {input_name: mag_input})[0]

                            # Flipped
                            mag_flipped = np.flip(mag_input, axis=-1)
                            pred2 = model.run([output_name], {input_name: mag_flipped})[0]
                            pred2 = np.flip(pred2, axis=-1)

                            # Average predictions
                            pred_magnitude = (pred1 + pred2) / 2
                        else:
                            pred_magnitude = model.run([output_name], {input_name: mag_input})[0]

                        # Reshape back
                        pred_magnitude = np.squeeze(pred_magnitude)
                        pred_magnitude = torch.from_numpy(pred_magnitude).float()

                        # Apply mask to original magnitude
                        if stem_name == 'other':
                            # For 'other', subtract drums and bass from original
                            mask = 1.0 - (self._get_mask('drums', magnitude) + self._get_mask('bass', magnitude))
                            mask = torch.clamp(mask, 0, 1)
                        else:
                            mask = pred_magnitude / (magnitude + 1e-7)
                            mask = torch.clamp(mask, 0, 1)

                        masked_magnitude = magnitude * mask

                        # Reconstruct complex STFT
                        masked_stft = masked_magnitude * torch.exp(1j * phase)

                        # iSTFT to get time domain signal
                        chunk_output = torch.istft(
                            masked_stft,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            window=window,
                            center=True,
                            length=chunk_size
                        )

                        # Add to output with windowing for overlap
                        if i == 0:
                            stems_dict[stem_name][:, start_idx:end_idx] = chunk_output.numpy()[:, :end_idx - start_idx]
                        else:
                            # Apply crossfade in overlap region
                            fade_length = overlap_size
                            fade_in = np.linspace(0, 1, fade_length)
                            fade_out = np.linspace(1, 0, fade_length)

                            # Crossfade overlap region
                            overlap_start = start_idx
                            overlap_end = start_idx + fade_length

                            stems_dict[stem_name][:, overlap_start:overlap_end] *= fade_out
                            stems_dict[stem_name][:, overlap_start:overlap_end] += chunk_output.numpy()[:,
                                                                                   :fade_length] * fade_in

                            # Add non-overlapping part
                            if end_idx > overlap_end:
                                stems_dict[stem_name][:, overlap_end:end_idx] = chunk_output.numpy()[:,
                                                                                fade_length:end_idx - start_idx]

            # Remove padding
            for stem_name in stems_dict.keys():
                stems_dict[stem_name] = stems_dict[stem_name][:, pad_size:-pad_size]

            # Denoise if enabled
            if self.config["uvr5"].get("denoise", True):
                if progress_callback:
                    progress_callback(0.88, "Applying denoising...")

                for stem_name in stems_dict.keys():
                    stems_dict[stem_name] = self._denoise(stems_dict[stem_name])

            # Post-processing
            if self.config["uvr5"].get("post_process", True):
                if progress_callback:
                    progress_callback(0.92, "Post-processing stems...")

                # Ensure sum of stems equals original
                total_reconstruction = np.zeros_like(audio_np)
                for stem in stems_dict.values():
                    total_reconstruction += stem

                # Calculate scaling factor
                scale_factor = np.sum(audio_np ** 2) / (np.sum(total_reconstruction ** 2) + 1e-7)
                scale_factor = np.sqrt(scale_factor)

                # Apply scaling
                for stem_name in stems_dict.keys():
                    stems_dict[stem_name] *= scale_factor

            # Convert back to torch tensors
            stems_torch = {}
            for stem_name, stem_np in stems_dict.items():
                stem_tensor = torch.from_numpy(stem_np).float()

                # Resample back to original sample rate if needed
                if process_sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(process_sr, sample_rate)
                    stem_tensor = resampler(stem_tensor)

                # Restore original scale
                if max_val > 0:
                    stem_tensor *= max_val

                stems_torch[stem_name] = stem_tensor

            if progress_callback:
                progress_callback(0.95, "Calculating quality metrics...")

            # Calculate metrics
            metrics = self._calculate_metrics(waveform, stems_torch)

            if progress_callback:
                progress_callback(1.0, "Separation complete!")

            processing_time = time.time() - start_time

            return {
                "stems": stems_torch,
                "source_names": list(stems_torch.keys()),
                "metrics": metrics,
                "processing_time": processing_time,
                "model": self.model_name,
                "quality_settings": {
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                    "tta": self.config["uvr5"].get("tta", False),
                    "denoise": self.config["uvr5"].get("denoise", True)
                }
            }

        except Exception as e:
            logger.error(f"UVR5 separation failed: {e}")
            raise

    def _denoise(self, audio: np.ndarray, threshold: float = 0.001) -> np.ndarray:
        """Apply spectral gating denoising."""
        # Simple spectral gating
        stft = np.fft.rfft(audio, axis=-1)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Calculate noise threshold
        noise_floor = np.percentile(magnitude, 10, axis=-1, keepdims=True)
        mask = magnitude > (noise_floor + threshold)

        # Apply mask
        magnitude_denoised = magnitude * mask
        stft_denoised = magnitude_denoised * np.exp(1j * phase)

        return np.fft.irfft(stft_denoised, n=audio.shape[-1], axis=-1)

    def _calculate_metrics(self, original: torch.Tensor, stems: Dict) -> Dict:
        """Calculate separation quality metrics."""
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

        # SNR calculation
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean((original - reconstructed) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        metrics["reconstruction_snr"] = float(snr)

        # SDR approximation
        sdr = 10 * torch.log10(signal_power / (torch.mean((original - reconstructed) ** 2) + 1e-10))
        metrics["source_distortion_ratio"] = float(sdr)

        metrics["num_stems"] = len(stems)

        # Energy distribution
        total_energy = torch.sum(original ** 2)
        for name, stem in stems.items():
            energy = torch.sum(stem ** 2)
            metrics[f"{name}_energy"] = float(energy)
            metrics[f"{name}_energy_ratio"] = float(energy / (total_energy + 1e-10))

        # Quality score
        metrics["overall_quality_score"] = min((float(snr) / 30.0) * 100, 100.0)

        return metrics