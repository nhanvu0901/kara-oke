"""
Optimized Audio Loader
======================
Production-ready audio loading with format support and preprocessing.
"""

import torch
import torchaudio
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Try to import additional audio libraries for broader format support
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    import numpy as np
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


class AudioLoader:
    """Optimized audio loader with multiple backend support."""

    # Supported formats by backend
    TORCHAUDIO_FORMATS = {'.wav', '.flac', '.ogg', '.opus'}
    SOUNDFILE_FORMATS = {'.wav', '.flac', '.ogg', '.mp3', '.m4a'}
    PYDUB_FORMATS = {'.mp3', '.m4a', '.aac', '.wma', '.wav', '.flac', '.ogg'}

    def __init__(self, config: Dict):
        """
        Initialize audio loader.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.target_sr = config.get("sample_rate", 44100)
        self.normalize = config.get("normalize_input", True)
        self.convert_to_stereo = config.get("convert_to_stereo", True)
        self.max_length = config.get("max_length_seconds", None)

        # Determine available backends
        self.backends = self._detect_backends()
        logger.info(f"Available audio backends: {', '.join(self.backends)}")

    def _detect_backends(self) -> list:
        """Detect available audio loading backends."""
        backends = ["torchaudio"]  # Always available with PyTorch

        if SOUNDFILE_AVAILABLE:
            backends.append("soundfile")
        if PYDUB_AVAILABLE:
            backends.append("pydub")

        return backends

    def load(self, file_path: Path) -> Dict:
        """
        Load and preprocess audio file.

        Args:
            file_path: Output file path
            format: Output format (inferred from extension if None)
            bits_per_sample: Bit depth for WAV files (16, 24, or 32)
        """
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine format
        if format is None:
            format = file_path.suffix[1:].lower()

        # Ensure 2D tensor
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0.99:
            waveform = waveform * (0.99 / max_val)

        # Save based on format
        if format == "wav":
            bits = bits_per_sample or self.config.get("output_bitdepth", 24)
            torchaudio.save(
                str(file_path),
                waveform,
                sample_rate,
                bits_per_sample=bits
            )
        else:
            torchaudio.save(str(file_path), waveform, sample_rate)

        logger.info(f"Saved: {file_path.name} ({format.upper()}, {sample_rate}Hz)")

    def get_info(self, file_path: Path) -> Dict:
        """
        Get audio file information without loading the full audio.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with file information
        """
        info = {}

        try:
            # Try to get info using torchaudio
            metadata = torchaudio.info(str(file_path))
            info = {
                "sample_rate": metadata.sample_rate,
                "channels": metadata.num_channels,
                "frames": metadata.num_frames,
                "duration": metadata.num_frames / metadata.sample_rate,
                "format": file_path.suffix[1:].lower(),
                "bits_per_sample": getattr(metadata, 'bits_per_sample', None),
                "encoding": getattr(metadata, 'encoding', None)
            }
        except Exception as e:
            logger.debug(f"Could not get info with torchaudio: {e}")

            # Try with soundfile if available
            if SOUNDFILE_AVAILABLE:
                try:
                    with sf.SoundFile(str(file_path)) as f:
                        info = {
                            "sample_rate": f.samplerate,
                            "channels": f.channels,
                            "frames": f.frames,
                            "duration": f.frames / f.samplerate,
                            "format": f.format,
                            "subtype": f.subtype
                        }
                except Exception as e:
                    logger.debug(f"Could not get info with soundfile: {e}")

        # Add file size
        info["file_size_mb"] = file_path.stat().st_size / (1024 * 1024)
        info["filename"] = file_path.name

        return info

    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate an audio file before processing.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if file exists
        if not file_path.exists():
            return False, f"File not found: {file_path}"

        # Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        max_size = self.config.get("max_file_size_mb", 2048)  # 2GB default
        if size_mb > max_size:
            return False, f"File too large: {size_mb:.1f}MB (max: {max_size}MB)"

        # Check format
        file_ext = file_path.suffix.lower()
        all_formats = self.TORCHAUDIO_FORMATS | self.SOUNDFILE_FORMATS | self.PYDUB_FORMATS
        if file_ext not in all_formats:
            return False, f"Unsupported format: {file_ext}"

        # Try to read file info
        try:
            info = self.get_info(file_path)
            if info.get("duration", 0) == 0:
                return False, "Audio file appears to be empty"
        except Exception as e:
            return False, f"Cannot read file info: {str(e)}"

        return True, None

    def batch_load(self, file_paths: list[Path], max_workers: int = 4) -> list[Dict]:
        """
        Load multiple audio files in parallel.

        Args:
            file_paths: List of file paths
            max_workers: Maximum number of parallel workers

        Returns:
            List of audio data dictionaries
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        failed = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.load, path): path
                for path in file_paths
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    data = future.result()
                    results.append(data)
                    logger.info(f"Loaded: {path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {path.name}: {e}")
                    failed.append((path, str(e)))

        if failed:
            logger.warning(f"Failed to load {len(failed)} files")

        return results

    def resample(self, waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """
        Resample audio to target sample rate.

        Args:
            waveform: Audio tensor
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio tensor
        """
        if orig_sr == target_sr:
            return waveform

        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr,
            resampling_method="kaiser_window",
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            beta=14.769656459379492
        )

        return resampler(waveform)

    def normalize_loudness(self, waveform: torch.Tensor, target_db: float = -14.0) -> torch.Tensor:
        """
        Normalize audio loudness to target dB (EBU R128 style).

        Args:
            waveform: Audio tensor
            target_db: Target loudness in dB

        Returns:
            Normalized audio tensor
        """
        # Calculate current loudness (simplified)
        rms = torch.sqrt(torch.mean(waveform ** 2))
        current_db = 20 * torch.log10(rms + 1e-10)

        # Calculate gain
        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20)

        # Apply gain with limiting
        normalized = waveform * gain
        max_val = torch.max(torch.abs(normalized))
        if max_val > 0.99:
            normalized = normalized * (0.99 / max_val)

        return normalized

    def split_channels(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Split stereo audio into separate channels.

        Args:
            waveform: Stereo audio tensor [2, samples]

        Returns:
            Dictionary with 'left', 'right', 'mid', 'side' channels
        """
        if waveform.shape[0] != 2:
            raise ValueError(f"Expected stereo audio, got {waveform.shape[0]} channels")

        left = waveform[0]
        right = waveform[1]
        mid = (left + right) / 2
        side = (left - right) / 2

        return {
            "left": left,
            "right": right,
            "mid": mid,
            "side": side
        }

    def merge_channels(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        Merge separate channels into stereo.

        Args:
            left: Left channel tensor
            right: Right channel tensor

        Returns:
            Stereo audio tensor [2, samples]
        """
        return torch.stack([left, right], dim=0)

    def trim_silence(self,
                    waveform: torch.Tensor,
                    threshold_db: float = -60.0,
                    min_silence_duration: float = 0.1,
                    sample_rate: int = 44100) -> torch.Tensor:
        """
        Trim silence from beginning and end of audio.

        Args:
            waveform: Audio tensor
            threshold_db: Silence threshold in dB
            min_silence_duration: Minimum silence duration in seconds
            sample_rate: Sample rate

        Returns:
            Trimmed audio tensor
        """
        # Convert threshold to linear
        threshold = 10 ** (threshold_db / 20)

        # Calculate frame-wise energy
        frame_length = int(min_silence_duration * sample_rate)

        if waveform.dim() > 1:
            # Use mono mix for detection
            mono = torch.mean(waveform, dim=0)
        else:
            mono = waveform

        # Find non-silent regions
        amplitude = torch.abs(mono)

        # Moving average filter for smoother detection
        kernel_size = frame_length
        if kernel_size > 1:
            kernel = torch.ones(kernel_size) / kernel_size
            amplitude_smooth = torch.nn.functional.conv1d(
                amplitude.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=kernel_size // 2
            ).squeeze()
        else:
            amplitude_smooth = amplitude

        # Find start and end points
        non_silent = amplitude_smooth > threshold
        indices = torch.where(non_silent)[0]

        if len(indices) == 0:
            # All silence, return original
            return waveform

        start_idx = indices[0].item()
        end_idx = indices[-1].item() + 1

        # Apply trimming
        if waveform.dim() > 1:
            return waveform[:, start_idx:end_idx]
        else:
            return waveform[start_idx:end_idx]

    def apply_fade(self,
                   waveform: torch.Tensor,
                   fade_in_duration: float = 0.01,
                   fade_out_duration: float = 0.01,
                   sample_rate: int = 44100) -> torch.Tensor:
        """
        Apply fade in/out to audio.

        Args:
            waveform: Audio tensor
            fade_in_duration: Fade in duration in seconds
            fade_out_duration: Fade out duration in seconds
            sample_rate: Sample rate

        Returns:
            Audio with fades applied
        """
        num_samples = waveform.shape[-1]

        # Calculate fade lengths
        fade_in_samples = min(int(fade_in_duration * sample_rate), num_samples // 2)
        fade_out_samples = min(int(fade_out_duration * sample_rate), num_samples // 2)

        # Create fade curves (linear)
        fade_in = torch.linspace(0, 1, fade_in_samples)
        fade_out = torch.linspace(1, 0, fade_out_samples)

        # Apply fades
        result = waveform.clone()

        if waveform.dim() > 1:
            # Multichannel
            result[:, :fade_in_samples] *= fade_in
            result[:, -fade_out_samples:] *= fade_out
        else:
            # Mono
            result[:fade_in_samples] *= fade_in
            result[-fade_out_samples:] *= fade_out

        return result

    def get_audio_stats(self, waveform: torch.Tensor, sample_rate: int) -> Dict:
        """
        Calculate comprehensive audio statistics.

        Args:
            waveform: Audio tensor
            sample_rate: Sample rate

        Returns:
            Dictionary of audio statistics
        """
        stats = {}

        # Basic stats
        stats["min"] = float(torch.min(waveform))
        stats["max"] = float(torch.max(waveform))
        stats["mean"] = float(torch.mean(waveform))
        stats["std"] = float(torch.std(waveform))

        # RMS and peak
        stats["rms"] = float(torch.sqrt(torch.mean(waveform ** 2)))
        stats["peak"] = float(torch.max(torch.abs(waveform)))
        stats["peak_db"] = float(20 * torch.log10(stats["peak"] + 1e-10))

        # Crest factor (peak/rms ratio)
        stats["crest_factor"] = stats["peak"] / (stats["rms"] + 1e-10)
        stats["crest_factor_db"] = float(20 * torch.log10(stats["crest_factor"] + 1e-10))

        # DC offset
        stats["dc_offset"] = float(torch.mean(waveform))

        # Dynamic range
        sorted_samples = torch.sort(torch.abs(waveform.flatten()))[0]
        percentile_95 = sorted_samples[int(len(sorted_samples) * 0.95)]
        percentile_10 = sorted_samples[int(len(sorted_samples) * 0.10)]
        stats["dynamic_range_db"] = float(20 * torch.log10(percentile_95 / (percentile_10 + 1e-10)))

        # Zero crossing rate
        if waveform.dim() > 1:
            mono = torch.mean(waveform, dim=0)
        else:
            mono = waveform
        zero_crossings = torch.sum(torch.diff(torch.sign(mono)) != 0)
        stats["zero_crossing_rate"] = float(zero_crossings) / len(mono)

        # Duration
        stats["duration_seconds"] = waveform.shape[-1] / sample_rate
        stats["num_samples"] = waveform.shape[-1]
        stats["sample_rate"] = sample_rate

        return stats Path to audio file

        Returns:
            Dictionary containing audio data and metadata
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        file_ext = file_path.suffix.lower()
        
        # Log loading
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"Loading: {file_path.name} ({file_size_mb:.1f}MB, format: {file_ext})")
        
        # Try loading with different backends
        waveform, sample_rate = None, None
        load_error = None
        
        # Try torchaudio first (fastest for supported formats)
        if file_ext in self.TORCHAUDIO_FORMATS:
            try:
                waveform, sample_rate = self._load_with_torchaudio(file_path)
            except Exception as e:
                load_error = e
                logger.debug(f"Torchaudio failed: {e}")
        
        # Try soundfile if available
        if waveform is None and SOUNDFILE_AVAILABLE and file_ext in self.SOUNDFILE_FORMATS:
            try:
                waveform, sample_rate = self._load_with_soundfile(file_path)
            except Exception as e:
                load_error = e
                logger.debug(f"Soundfile failed: {e}")
        
        # Try pydub as fallback (supports most formats but slower)
        if waveform is None and PYDUB_AVAILABLE:
            try:
                waveform, sample_rate = self._load_with_pydub(file_path)
            except Exception as e:
                load_error = e
                logger.debug(f"Pydub failed: {e}")
        
        # Final torchaudio attempt for any format
        if waveform is None:
            try:
                waveform, sample_rate = self._load_with_torchaudio(file_path)
            except Exception as e:
                load_error = e
        
        if waveform is None:
            raise RuntimeError(f"Failed to load audio file: {load_error}")
        
        # Preprocess audio
        waveform, sample_rate = self._preprocess(waveform, sample_rate)
        
        # Calculate metadata
        duration = waveform.shape[-1] / sample_rate
        channels = waveform.shape[0] if waveform.dim() > 1 else 1
        
        # Check for potential issues
        max_amplitude = torch.max(torch.abs(waveform)).item()
        is_clipped = max_amplitude > 0.99
        
        if is_clipped:
            logger.warning(f"Audio may be clipped (max amplitude: {max_amplitude:.3f})")
        
        # DC offset check
        dc_offset = torch.mean(waveform).item()
        if abs(dc_offset) > 0.01:
            logger.info(f"DC offset detected: {dc_offset:.4f}, removing...")
            waveform = waveform - dc_offset
        
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "duration": duration,
            "channels": channels,
            "samples": waveform.shape[-1],
            "format": file_ext[1:],
            "filename": file_path.stem,
            "filepath": str(file_path),
            "max_amplitude": max_amplitude,
            "is_clipped": is_clipped,
            "file_size_mb": file_size_mb
        }
    
    def _load_with_torchaudio(self, file_path: Path) -> Tuple[torch.Tensor, int]:
        """Load audio using torchaudio."""
        waveform, sample_rate = torchaudio.load(str(file_path))
        return waveform, sample_rate
    
    def _load_with_soundfile(self, file_path: Path) -> Tuple[torch.Tensor, int]:
        """Load audio using soundfile."""
        data, sample_rate = sf.read(str(file_path), always_2d=True)
        # Soundfile returns (samples, channels), convert to (channels, samples)
        waveform = torch.from_numpy(data.T).float()
        return waveform, sample_rate
    
    def _load_with_pydub(self, file_path: Path) -> Tuple[torch.Tensor, int]:
        """Load audio using pydub."""
        audio = AudioSegment.from_file(str(file_path))
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())
        
        # Reshape based on channels
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).T
        else:
            samples = samples.reshape(1, -1)
        
        # Convert to float32 and normalize
        max_val = 2 ** (audio.sample_width * 8 - 1)
        samples = samples.astype(np.float32) / max_val
        
        waveform = torch.from_numpy(samples)
        sample_rate = audio.frame_rate
        
        return waveform, sample_rate
    
    def _preprocess(self, waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, int]:
        """Preprocess audio for optimal separation."""
        # Ensure 2D tensor [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Limit length if specified
        if self.max_length and waveform.shape[-1] > self.max_length * sample_rate:
            max_samples = int(self.max_length * sample_rate)
            logger.info(f"Truncating audio to {self.max_length} seconds")
            waveform = waveform[:, :max_samples]

        # Resample if needed
        if sample_rate != self.target_sr:
            logger.info(f"Resampling from {sample_rate}Hz to {self.target_sr}Hz")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sr,
                resampling_method="kaiser_window"
            )
            waveform = resampler(waveform)
            sample_rate = self.target_sr

        # Convert to stereo if requested and not already
        if self.convert_to_stereo:
            if waveform.shape[0] == 1:
                # Mono to stereo
                waveform = waveform.repeat(2, 1)
                logger.debug("Converted mono to stereo")
            elif waveform.shape[0] > 2:
                # Multichannel to stereo
                waveform = waveform[:2, :]
                logger.debug(f"Reduced {waveform.shape[0]} channels to stereo")

        # Normalize if requested
        if self.normalize:
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                target_peak = 0.95  # Leave some headroom
                waveform = waveform * (target_peak / max_val)
                logger.debug(f"Normalized audio (peak: {max_val:.3f} -> {target_peak})")

        return waveform, sample_rate

    def save(self,
             waveform: torch.Tensor,
             sample_rate: int,
             file_path: Path,
             format: Optional[str] = None,
             bits_per_sample: Optional[int] = None) -> None:
        """
        Save audio to file.

        Args:
            waveform: Audio tensor [channels, samples]
            sample_rate: Sample rate in Hz
            file_path: Output file path
            format: Output format (inferred from extension if None)
            bits_per_sample: Bit depth for WAV files (16, 24, or 32)
        """