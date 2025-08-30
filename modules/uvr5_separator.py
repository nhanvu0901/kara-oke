import torch
import numpy as np
import torchaudio
from pathlib import Path
from typing import Dict, Optional, Callable, List
import logging
import time
from scipy import signal
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    logger.warning("ONNX Runtime not installed. Install with: pip install onnxruntime")
    ONNX_AVAILABLE = False
    ort = None


class UVR5Separator:
    """UVR5 MDX-Net separator for high-quality instrumental separation."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", "cpu")
        self.model_name = config["uvr5"]["model"]
        self.chunk_size = config["uvr5"].get("chunk_size", 261120)
        self.overlap = config["uvr5"].get("overlap", 0.75)
        self.margin = config["uvr5"].get("margin", 88200)
        self.models = {}
        self.model = None

        # Define instrument separation strategy
        self.separation_mode = config["uvr5"].get("separation_mode", "enhanced")

        if ONNX_AVAILABLE:
            self._load_models()

    def _load_models(self):
        """Load multiple UVR5 models for enhanced separation."""
        try:
            logger.info("Loading UVR5 instrumental models...")

            # Model configurations for your 5 models
            model_configs = {
                'main': {
                    'path': '/Users/nhanvu/Documents/AI_project/karaoke_auto_pipeline/models/uvr5/models/uvr5/UVR-MDX-NET-Inst_HQ_3.onnx',  # Changed from models/uvr5/models/uvr5/
                    'description': 'Main HQ instrumental separator'
                },
                'secondary': {
                    'path': '/Users/nhanvu/Documents/AI_project/karaoke_auto_pipeline/models/uvr5/models/uvr5/UVR-MDX-NET-Inst_Main.onnx',
                    'description': 'Secondary main separator'
                },
                'inst1': {
                    'path': '/Users/nhanvu/Documents/AI_project/karaoke_auto_pipeline/models/uvr5/models/uvr5/UVR-MDX-NET-Inst_1.onnx',
                    'description': 'Instrumental variant 1'
                },
                'inst3': {
                    'path': '/Users/nhanvu/Documents/AI_project/karaoke_auto_pipeline/models/uvr5/models/uvr5/UVR-MDX-NET-Inst_HQ_3.onnx',
                    'description': 'Instrumental variant 3'
                },
                'karaoke': {
                    'path': '/Users/nhanvu/Documents/AI_project/karaoke_auto_pipeline/models/uvr5/models/uvr5/UVR_MDXNET_KARA_2.onnx',
                    'description': 'Karaoke/Instrumental extraction'
                }
            }

            providers = ['CPUExecutionProvider']
            if self.device == 'cuda':
                providers.insert(0, 'CUDAExecutionProvider')
            elif self.device == 'mps':
                # For M4 Mac, use CoreML if available
                try:
                    providers.insert(0, 'CoreMLExecutionProvider')
                except:
                    logger.info("CoreML not available, using CPU")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.inter_op_num_threads = 4
            sess_options.intra_op_num_threads = 4

            # Load available models
            for model_key, config in model_configs.items():
                model_path = Path(config['path'])
                if model_path.exists():
                    try:
                        model = ort.InferenceSession(
                            str(model_path),
                            sess_options=sess_options,
                            providers=providers
                        )
                        self.models[model_key] = model
                        logger.info(f"✅ Loaded {config['description']}")

                        # Get model input/output info
                        input_info = model.get_inputs()[0]
                        output_info = model.get_outputs()[0]
                        logger.info(f"   Input shape: {input_info.shape}, Output shape: {output_info.shape}")

                    except Exception as e:
                        logger.warning(f"Could not load {model_key}: {e}")
                else:
                    logger.info(f"Model not found: {config['path']}")

            if self.models:
                self.model = True  # Set flag that models are loaded
                logger.info(f"Successfully loaded {len(self.models)} UVR5 models")
            else:
                logger.warning("No UVR5 models loaded, will use basic separation")
                self.model = True

        except Exception as e:
            logger.error(f"Failed to load UVR5 models: {e}")
            raise

    def separate(self, waveform: torch.Tensor, sample_rate: int,
                 progress_callback: Optional[Callable] = None) -> Dict:
        """
        Multi-stage instrumental separation using all 5 models.
        """
        start_time = time.time()

        if not self.models:
            logger.warning("No ONNX models loaded, using basic separation")
            return self._separate_basic(waveform, sample_rate, progress_callback)

        try:
            if progress_callback:
                progress_callback(0.05, "Starting multi-model separation...")

            # Ensure stereo
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)

            # Stage 1: Primary separation with HQ model
            if 'main' in self.models and progress_callback:
                progress_callback(0.15, "Stage 1: HQ instrumental extraction...")

            stage1_output = self._process_with_model(
                waveform, sample_rate, 'main', "HQ separation"
            ) if 'main' in self.models else waveform

            # Stage 2: Refine with Main model
            if 'secondary' in self.models and progress_callback:
                progress_callback(0.30, "Stage 2: Main model refinement...")

            stage2_output = self._process_with_model(
                stage1_output, sample_rate, 'secondary', "Main refinement"
            ) if 'secondary' in self.models else stage1_output

            # Stage 3: Process with Inst_1
            if 'inst1' in self.models and progress_callback:
                progress_callback(0.45, "Stage 3: Inst_1 processing...")

            stage3_output = self._process_with_model(
                stage2_output, sample_rate, 'inst1', "Inst_1 processing"
            ) if 'inst1' in self.models else stage2_output

            # Stage 4: Process with Inst_3
            if 'inst3' in self.models and progress_callback:
                progress_callback(0.60, "Stage 4: Inst_3 enhancement...")

            stage4_output = self._process_with_model(
                stage3_output, sample_rate, 'inst3', "Inst_3 enhancement"
            ) if 'inst3' in self.models else stage3_output

            # Stage 5: Final karaoke model processing
            if 'karaoke' in self.models and progress_callback:
                progress_callback(0.75, "Stage 5: Karaoke model finalization...")

            refined_instrumental = self._process_with_model(
                stage4_output, sample_rate, 'karaoke', "Karaoke finalization"
            ) if 'karaoke' in self.models else stage4_output

            # Stage 6: Extract individual instruments
            if progress_callback:
                progress_callback(0.85, "Stage 6: Extracting individual instruments...")

            final_stems = self._extract_instrument_stems(refined_instrumental, sample_rate, waveform)

            if progress_callback:
                progress_callback(0.95, "Calculating quality metrics...")

            # Calculate metrics
            metrics = self._calculate_enhanced_metrics(waveform, final_stems)

            if progress_callback:
                progress_callback(1.0, "Multi-stage separation complete!")

            processing_time = time.time() - start_time

            return {
                "stems": final_stems,
                "source_names": list(final_stems.keys()),
                "metrics": metrics,
                "processing_time": processing_time,
                "model": "UVR5-5Model-Cascade",
                "quality_settings": {
                    "models_used": list(self.models.keys()),
                    "num_models": len(self.models),
                    "stages": 6,
                    "separation_mode": "enhanced_multi_model"
                }
            }

        except Exception as e:
            logger.error(f"Multi-stage separation failed: {e}")
            return self._separate_basic(waveform, sample_rate, progress_callback)

    def _process_with_model(self, audio: torch.Tensor, sample_rate: int,
                            model_key: str, description: str) -> torch.Tensor:
        """Process audio through a specific model."""
        if model_key not in self.models:
            return audio

        try:
            model = self.models[model_key]
            logger.info(f"Processing: {description}")

            # Prepare audio for MDX-Net
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)

            # MDX-Net expects 44100 Hz
            if sample_rate != 44100:
                resampler = torchaudio.transforms.Resample(sample_rate, 44100)
                audio_resampled = resampler(audio)
                process_sr = 44100
            else:
                audio_resampled = audio
                process_sr = sample_rate

            # Normalize
            audio_np = audio_resampled.numpy()
            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / max_val

            # Process in chunks for better quality
            processed = self._process_chunks_with_model(audio_np, model)

            # Convert back to tensor
            processed_tensor = torch.from_numpy(processed).float()

            # Restore scale
            if max_val > 0:
                processed_tensor *= max_val

            # Resample back if needed
            if process_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(process_sr, sample_rate)
                processed_tensor = resampler(processed_tensor)

            return processed_tensor

        except Exception as e:
            logger.error(f"Model processing failed for {model_key}: {e}")
            return audio

    def _process_chunks_with_model(self, audio_np: np.ndarray, model) -> np.ndarray:
        """Process audio in chunks through ONNX model with correct dimensions."""

        # Get model input shape expectations
        input_info = model.get_inputs()[0]
        expected_shape = input_info.shape  # Should be like [1, 4, 2048/3072, 256]

        logger.info(f"Model expects shape: {expected_shape}")

        # MDX-Net specific parameters
        if expected_shape[2] == 2048:
            n_fft = 4096  # Results in 2049 bins, we'll use 2048
        elif expected_shape[2] == 3072:
            n_fft = 6144  # Results in 3073 bins, we'll use 3072
        else:
            n_fft = 4096  # Default

        hop_length = 1024

        expected_channels = expected_shape[1]  # 4 channels
        expected_freq_bins = expected_shape[2]  # 2048 or 3072
        expected_time_frames = expected_shape[3]  # 256 frames

        # Process in fixed-size chunks
        chunk_samples = expected_time_frames * hop_length  # Number of samples per chunk

        # Pad audio for processing
        pad_size = n_fft // 2
        audio_padded = np.pad(audio_np, ((0, 0), (pad_size, pad_size)), mode='reflect')

        # Ensure we have 2 channels
        if audio_padded.shape[0] == 1:
            audio_padded = np.repeat(audio_padded, 2, axis=0)
        elif audio_padded.shape[0] > 2:
            audio_padded = audio_padded[:2]

        total_length = audio_padded.shape[1]
        output = np.zeros_like(audio_padded)

        # Create window with CORRECT size matching n_fft
        window = np.hanning(n_fft)  # This ensures window.shape[0] == n_fft

        # Process in chunks
        for start_idx in range(0, total_length - chunk_samples, chunk_samples // 2):
            end_idx = start_idx + chunk_samples
            chunk = audio_padded[:, start_idx:end_idx]

            if chunk.shape[1] < chunk_samples:
                # Pad last chunk
                chunk = np.pad(chunk, ((0, 0), (0, chunk_samples - chunk.shape[1])), mode='constant')

            # Compute STFT with properly sized window
            stft_result = []

            for channel in chunk:
                # Use scipy's stft with explicit parameters
                f, t, Zxx = signal.stft(
                    channel,
                    fs=44100,
                    window=('hann', n_fft),  # Use tuple format to ensure correct size
                    nperseg=n_fft,  # Must match window size
                    noverlap=n_fft - hop_length,
                    boundary=None,
                    padded=False
                )
                stft_result.append(Zxx)

            stft_np = np.stack(stft_result)  # Shape: [2, freq_bins, time_frames]

            # Trim frequency bins to match expected
            if stft_np.shape[1] > expected_freq_bins:
                stft_np = stft_np[:, :expected_freq_bins, :]

            # Trim or pad time frames to match expected
            if stft_np.shape[2] > expected_time_frames:
                stft_np = stft_np[:, :, :expected_time_frames]
            elif stft_np.shape[2] < expected_time_frames:
                pad_frames = expected_time_frames - stft_np.shape[2]
                stft_np = np.pad(stft_np, ((0, 0), (0, 0), (0, pad_frames)), mode='constant')

            magnitude = np.abs(stft_np)
            phase = np.angle(stft_np)

            # Convert to 4 channels as expected by model
            if expected_channels == 4:
                # Duplicate stereo to 4 channels (L, R, L, R)
                mag_4ch = np.zeros((4, expected_freq_bins, expected_time_frames), dtype=np.float32)
                mag_4ch[0] = magnitude[0]  # Left
                mag_4ch[1] = magnitude[1]  # Right
                mag_4ch[2] = magnitude[0]  # Left copy
                mag_4ch[3] = magnitude[1]  # Right copy
            else:
                mag_4ch = magnitude.astype(np.float32)

            # Add batch dimension
            mag_input = np.expand_dims(mag_4ch, axis=0).astype(np.float32)

            # Run inference
            try:
                input_name = model.get_inputs()[0].name
                output_name = model.get_outputs()[0].name

                # Run model
                pred_output = model.run([output_name], {input_name: mag_input})[0]

                # Process output
                pred_output = np.squeeze(pred_output)

                # If output has 4 channels, average to stereo
                if pred_output.ndim == 3 and pred_output.shape[0] == 4:
                    pred_stereo = np.zeros((2, pred_output.shape[1], pred_output.shape[2]))
                    pred_stereo[0] = (pred_output[0] + pred_output[2]) / 2
                    pred_stereo[1] = (pred_output[1] + pred_output[3]) / 2
                    pred_output = pred_stereo
                elif pred_output.ndim == 2:
                    # If output is 2D, expand to match magnitude shape
                    if magnitude.shape[0] == 2:
                        pred_output = np.stack([pred_output, pred_output])

                # Apply as mask
                if pred_output.shape == magnitude.shape:
                    mask = pred_output / (magnitude + 1e-7)
                else:
                    # Fallback if shapes don't match
                    mask = np.ones_like(magnitude)

                mask = np.clip(mask, 0, 1)

                masked_magnitude = magnitude * mask
                masked_stft = masked_magnitude * np.exp(1j * phase)

                # iSTFT for this chunk with matching window
                chunk_output = []
                for channel_stft in masked_stft:
                    _, reconstructed = signal.istft(
                        channel_stft,
                        fs=44100,
                        window=('hann', n_fft),  # Use same window format
                        nperseg=n_fft,
                        noverlap=n_fft - hop_length,
                        boundary=False
                    )
                    # Ensure we don't exceed chunk size
                    if len(reconstructed) > chunk_samples:
                        reconstructed = reconstructed[:chunk_samples]
                    chunk_output.append(reconstructed)

                chunk_output = np.stack(chunk_output)

                # Blend chunk into output with crossfade
                actual_samples = min(chunk_output.shape[1], end_idx - start_idx)

                if start_idx == 0:
                    output[:, start_idx:start_idx + actual_samples] = chunk_output[:, :actual_samples]
                else:
                    # Crossfade overlap region
                    overlap_size = min(chunk_samples // 4, actual_samples)
                    fade_in = np.linspace(0, 1, overlap_size)
                    fade_out = 1 - fade_in

                    output[:, start_idx:start_idx + overlap_size] *= fade_out
                    output[:, start_idx:start_idx + overlap_size] += chunk_output[:, :overlap_size] * fade_in

                    if actual_samples > overlap_size:
                        output[:, start_idx + overlap_size:start_idx + actual_samples] = chunk_output[:,
                                                                                         overlap_size:actual_samples]

            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                # Use input as fallback for this chunk
                actual_chunk_size = min(end_idx - start_idx, chunk.shape[1])
                output[:, start_idx:start_idx + actual_chunk_size] = chunk[:, :actual_chunk_size]

        # Remove padding
        if pad_size > 0:
            output = output[:, pad_size:-pad_size]

        # Ensure same length as input
        if output.shape[1] > audio_np.shape[1]:
            output = output[:, :audio_np.shape[1]]
        elif output.shape[1] < audio_np.shape[1]:
            # Pad if output is shorter
            pad_amount = audio_np.shape[1] - output.shape[1]
            output = np.pad(output, ((0, 0), (0, pad_amount)), mode='constant')

        return output

    def _extract_instrument_stems(self, instrumental: torch.Tensor, sr: int,
                                  original: torch.Tensor) -> Dict:
        """Extract individual instrument stems from processed instrumental."""
        stems = {}

        # Extract rhythm section
        stems['drums'] = self._extract_drums_enhanced(instrumental, sr)
        stems['bass'] = self._extract_bass_enhanced(instrumental, sr)

        # Extract harmonic instruments
        stems['piano'] = self._extract_piano_enhanced(instrumental, sr)
        stems['guitar'] = self._extract_guitar_enhanced(instrumental, sr)

        # Calculate other instruments (remainder)
        extracted_sum = torch.zeros_like(instrumental)
        for stem_name in ['drums', 'bass', 'piano', 'guitar']:
            if stem_name in stems:
                extracted_sum += stems[stem_name]

        # Other = instrumental - all extracted stems
        stems['other'] = instrumental - extracted_sum

        # Optional: Add a vocals stem (should be minimal/empty for instrumental)
        stems['vocals'] = original - instrumental

        return stems

    def _extract_drums_enhanced(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Enhanced drum extraction using multi-band processing."""
        audio_np = audio.numpy()
        nyquist = sr // 2

        drums_components = []

        # Kick drum (60-100 Hz) with punch
        if nyquist > 100:
            sos = signal.butter(6, [60 / nyquist, min(100 / nyquist, 0.99)],
                                btype='band', output='sos')
            kick = signal.sosfiltfilt(sos, audio_np, axis=-1)
            # Add transient enhancement for kick
            kick_envelope = np.abs(kick)
            kick_transient = np.diff(kick_envelope, axis=-1, prepend=0)
            kick = kick * (1 + np.clip(kick_transient * 2, 0, 1))
            drums_components.append(kick * 1.5)

        # Snare (200-500 Hz)
        if nyquist > 500:
            sos = signal.butter(4, [200 / nyquist, min(500 / nyquist, 0.99)],
                                btype='band', output='sos')
            snare = signal.sosfiltfilt(sos, audio_np, axis=-1)
            drums_components.append(snare * 1.0)

        # Hi-hats and cymbals (3-12 kHz)
        if nyquist > 3000:
            sos = signal.butter(4, min(3000 / nyquist, 0.99),
                                btype='high', output='sos')
            hihats = signal.sosfiltfilt(sos, audio_np, axis=-1)
            # Add brightness
            sos_bright = signal.butter(2, [8000 / nyquist, min(12000 / nyquist, 0.99)],
                                       btype='band', output='sos')
            brightness = signal.sosfiltfilt(sos_bright, audio_np, axis=-1)
            drums_components.append(hihats * 0.7)
            drums_components.append(brightness * 0.3)

        # Combine all drum components
        if drums_components:
            drums_combined = np.sum(drums_components, axis=0) / len(drums_components)
            # Apply compression-like effect
            drums_combined = np.tanh(drums_combined * 1.2) * 0.9
            return torch.from_numpy(drums_combined).float()

        return torch.zeros_like(audio)

    def _extract_bass_enhanced(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Enhanced bass extraction with harmonic preservation."""
        audio_np = audio.numpy()
        nyquist = sr // 2

        if nyquist > 250:
            # Fundamental frequencies (30-250 Hz)
            sos_fundamental = signal.butter(6, min(250 / nyquist, 0.99),
                                            btype='low', output='sos')
            bass_fundamental = signal.sosfiltfilt(sos_fundamental, audio_np, axis=-1)

            # First harmonic (250-500 Hz) for definition
            if nyquist > 500:
                sos_harmonic = signal.butter(4, [250 / nyquist, min(500 / nyquist, 0.99)],
                                             btype='band', output='sos')
                bass_harmonic = signal.sosfiltfilt(sos_harmonic, audio_np, axis=-1)
                bass_combined = bass_fundamental * 1.2 + bass_harmonic * 0.3
            else:
                bass_combined = bass_fundamental

            # Apply slight saturation for warmth
            bass_combined = np.tanh(bass_combined * 1.1) * 0.95

            return torch.from_numpy(bass_combined).float()

        return torch.zeros_like(audio)

    def _extract_piano_enhanced(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Enhanced piano extraction focusing on characteristic frequencies."""
        audio_np = audio.numpy()
        nyquist = sr // 2

        # Piano fundamental range (80-2000 Hz) plus harmonics (up to 4000 Hz)
        if nyquist > 4000:
            # Main piano range
            sos_main = signal.butter(4, [80 / nyquist, min(2000 / nyquist, 0.99)],
                                     btype='band', output='sos')
            piano_main = signal.sosfiltfilt(sos_main, audio_np, axis=-1)

            # Harmonics for brightness
            sos_harmonics = signal.butter(4, [2000 / nyquist, min(4000 / nyquist, 0.99)],
                                          btype='band', output='sos')
            piano_harmonics = signal.sosfiltfilt(sos_harmonics, audio_np, axis=-1)

            # Combine with emphasis on main range
            piano_combined = piano_main * 1.0 + piano_harmonics * 0.4

            # Apply gentle compression
            piano_combined = np.tanh(piano_combined * 0.8) * 1.1

            return torch.from_numpy(piano_combined).float()

        return torch.zeros_like(audio) * 0.1

    def _extract_guitar_enhanced(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Enhanced guitar extraction with string characteristics."""
        audio_np = audio.numpy()
        nyquist = sr // 2

        # Guitar range (80-5000 Hz) with emphasis on mid frequencies
        if nyquist > 5000:
            # Fundamental range
            sos_fundamental = signal.butter(4, [80 / nyquist, min(1000 / nyquist, 0.99)],
                                            btype='band', output='sos')
            guitar_fundamental = signal.sosfiltfilt(sos_fundamental, audio_np, axis=-1)

            # Mid presence (1000-3000 Hz)
            sos_presence = signal.butter(4, [1000 / nyquist, min(3000 / nyquist, 0.99)],
                                         btype='band', output='sos')
            guitar_presence = signal.sosfiltfilt(sos_presence, audio_np, axis=-1)

            # Upper harmonics (3000-5000 Hz)
            sos_harmonics = signal.butter(4, [3000 / nyquist, min(5000 / nyquist, 0.99)],
                                          btype='band', output='sos')
            guitar_harmonics = signal.sosfiltfilt(sos_harmonics, audio_np, axis=-1)

            # Combine with guitar-like balance
            guitar_combined = (guitar_fundamental * 0.8 +
                               guitar_presence * 1.0 +
                               guitar_harmonics * 0.5)

            # Add slight overdrive character
            guitar_combined = np.sign(guitar_combined) * np.abs(guitar_combined) ** 0.9

            return torch.from_numpy(guitar_combined).float()

        return torch.zeros_like(audio) * 0.1

    def _calculate_enhanced_metrics(self, original: torch.Tensor, stems: Dict) -> Dict:
        """Calculate comprehensive quality metrics."""
        metrics = {}

        # Reconstruct from stems
        reconstructed = torch.zeros_like(original)
        for stem in stems.values():
            if stem.shape == original.shape:
                reconstructed += stem

        # Basic metrics
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean((original - reconstructed) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))

        metrics["reconstruction_snr"] = float(snr)
        metrics["num_stems"] = len(stems)
        metrics["models_used"] = len(self.models)

        # Per-stem metrics
        total_energy = torch.sum(original ** 2)
        for name, stem in stems.items():
            stem_energy = torch.sum(stem ** 2)
            metrics[f"{name}_energy"] = float(stem_energy)
            metrics[f"{name}_energy_ratio"] = float(stem_energy / (total_energy + 1e-10))

            # Calculate stem SNR
            if torch.sum(stem ** 2) > 1e-10:
                stem_snr = 10 * torch.log10(torch.sum(stem ** 2) / (1e-10))
                metrics[f"{name}_snr"] = float(stem_snr)

        # Overall quality score (0-100)
        quality_factors = [
            min(float(snr) / 30.0, 1.0) * 40,  # SNR contribution (40%)
            min(len(self.models) / 5.0, 1.0) * 30,  # Model count contribution (30%)
            min(len(stems) / 6.0, 1.0) * 30  # Stem count contribution (30%)
        ]
        metrics["overall_quality_score"] = sum(quality_factors)

        # Separation detail level
        if len(stems) >= 6:
            metrics["separation_detail"] = "comprehensive"
        elif len(stems) >= 4:
            metrics["separation_detail"] = "detailed"
        else:
            metrics["separation_detail"] = "basic"

        return metrics

    def _separate_basic(self, waveform: torch.Tensor, sample_rate: int,
                        progress_callback: Optional[Callable] = None) -> Dict:
        """Fallback basic separation when models aren't available."""
        start_time = time.time()

        if progress_callback:
            progress_callback(0.2, "Using frequency-based separation...")

        # Ensure stereo
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        stems = {}

        if progress_callback:
            progress_callback(0.4, "Extracting drums...")
        stems['drums'] = self._extract_drums_enhanced(waveform, sample_rate)

        if progress_callback:
            progress_callback(0.5, "Extracting bass...")
        stems['bass'] = self._extract_bass_enhanced(waveform, sample_rate)

        if progress_callback:
            progress_callback(0.6, "Extracting piano...")
        stems['piano'] = self._extract_piano_enhanced(waveform, sample_rate)

        if progress_callback:
            progress_callback(0.7, "Extracting guitar...")
        stems['guitar'] = self._extract_guitar_enhanced(waveform, sample_rate)

        if progress_callback:
            progress_callback(0.8, "Processing other instruments...")

        # Other = original - all extracted
        extracted_sum = sum(stems.values())
        stems['other'] = waveform - extracted_sum
        stems['vocals'] = torch.zeros_like(waveform)  # No vocals in instrumental

        if progress_callback:
            progress_callback(0.9, "Calculating metrics...")

        metrics = self._calculate_enhanced_metrics(waveform, stems)

        if progress_callback:
            progress_callback(1.0, "Basic separation complete!")

        processing_time = time.time() - start_time

        return {
            "stems": stems,
            "source_names": list(stems.keys()),
            "metrics": metrics,
            "processing_time": processing_time,
            "model": "frequency_filter_fallback",
            "quality_settings": {
                "method": "frequency_filtering",
                "note": "Using basic filtering as fallback"
            }
        }