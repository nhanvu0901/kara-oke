import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable, Tuple
import logging
import time
from scipy import signal
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)
try:
    from demucs import pretrained
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
except ImportError:
    logger.warning("Demucs not installed. Install with: pip install demucs")
    pretrained = None


class EnhancedDemucsSeparator:
    """Enhanced Demucs separator with multi-model approach for better 'other' stem."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", "cpu")
        self.model_name = config["demucs"]["model"]
        self.shifts = config["demucs"].get("shifts", 20)  # Increased default
        self.overlap = config["demucs"].get("overlap", 0.85)  # Increased default
        self.models = {}  # Multiple models for better separation

        # Stems to keep (no vocals) - focus on 'other' quality
        self.target_stems = ["drums", "bass", "other"]

        # Load multiple models for ensemble approach
        if pretrained:
            self._load_models()

    def _load_models(self):
        """Load multiple Demucs models for ensemble approach."""
        try:
            # Primary model - htdemucs_6s for detailed separation
            logger.info(f"Loading primary model: {self.model_name}")
            self.models['primary'] = pretrained.get_model(self.model_name)
            self.models['primary'].to(self.device)
            self.models['primary'].eval()

            # Secondary model - htdemucs_ft for different characteristics
            if self.model_name != "htdemucs_ft":
                logger.info("Loading secondary model: htdemucs_ft")
                self.models['secondary'] = pretrained.get_model("htdemucs_ft")
                self.models['secondary'].to(self.device)
                self.models['secondary'].eval()

            # Use primary as main model for compatibility
            self.model = self.models['primary']

            logger.info(f"Models loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def separate(self, waveform: torch.Tensor, sample_rate: int,
                 progress_callback: Optional[Callable] = None) -> Dict:
        """
        Enhanced separation with focus on 'other' stem quality using multi-model approach.
        """
        if not self.models:
            raise RuntimeError("Models not loaded")

        start_time = time.time()

        # ==== PRE-PROCESSING ====
        if progress_callback:
            progress_callback(0.05, "Pre-processing audio...")

        preprocessed, original_peak = self._preprocess_audio(waveform, sample_rate)

        # ==== FREQUENCY BAND SEPARATION FOR 'OTHER' ====
        if progress_callback:
            progress_callback(0.10, "Preparing frequency bands for enhanced separation...")

        # Split into frequency bands for better 'other' stem
        bands = self._split_frequency_bands(preprocessed, sample_rate)

        # ==== MULTI-MODEL SEPARATION ====
        if progress_callback:
            progress_callback(0.15, "Running multi-model separation...")

        # Run primary model on full spectrum
        primary_stems = self._run_separation_pass(
            preprocessed, sample_rate,
            model_key='primary',
            shifts=self.shifts,
            overlap=self.overlap,
            progress_callback=progress_callback,
            progress_offset=0.15,
            progress_scale=0.25
        )

        # Run secondary model if available for comparison
        secondary_stems = {}
        if 'secondary' in self.models:
            if progress_callback:
                progress_callback(0.40, "Running secondary model for ensemble...")

            secondary_stems = self._run_separation_pass(
                preprocessed, sample_rate,
                model_key='secondary',
                shifts=self.shifts // 2,  # Faster for secondary
                overlap=self.overlap * 0.9,
                progress_callback=progress_callback,
                progress_offset=0.40,
                progress_scale=0.15
            )

        # ==== SPECIALIZED 'OTHER' PROCESSING ====
        if progress_callback:
            progress_callback(0.55, "Enhanced processing for instrumental stems...")

        # Process 'other' stem with specialized approach
        enhanced_other = self._enhance_other_stem(
            preprocessed, primary_stems, secondary_stems, bands, sample_rate
        )

        # Replace 'other' stem with enhanced version
        primary_stems['other'] = enhanced_other

        # ==== HARMONIC/PERCUSSIVE SEPARATION FOR 'OTHER' ====
        if progress_callback:
            progress_callback(0.70, "Separating harmonic and percussive components...")

        # Further split 'other' into sub-components
        other_components = self._separate_other_components(
            primary_stems['other'], sample_rate
        )

        # ==== REFINEMENT PASS ====
        if progress_callback:
            progress_callback(0.80, "Refining separation with residual analysis...")

        refined_stems = self._refine_separation_focused(
            preprocessed, primary_stems, other_components, sample_rate
        )

        # ==== POST-PROCESSING ====
        if progress_callback:
            progress_callback(0.90, "Post-processing stems...")

        final_stems = self._postprocess_stems_enhanced(
            refined_stems, preprocessed, sample_rate, original_peak
        )

        # ==== QUALITY METRICS ====
        if progress_callback:
            progress_callback(0.95, "Calculating metrics...")

        metrics = self._calculate_enhanced_metrics(preprocessed, final_stems)

        if progress_callback:
            progress_callback(1.0, "Complete!")

        return {
            "stems": final_stems,
            "source_names": list(final_stems.keys()),
            "metrics": metrics,
            "processing_time": time.time() - start_time,
            "model": self.model_name,
            "quality_settings": {
                "shifts": self.shifts,
                "overlap": self.overlap,
                "multi_model": len(self.models) > 1,
                "enhanced_other": True,
                "sub_components": len(other_components)
            }
        }

    def _preprocess_audio(self, waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, float]:
        """Pre-process audio for optimal separation."""
        audio = waveform.clone()

        # Store original peak for later restoration
        original_peak = torch.max(torch.abs(audio)).item()

        # 1. Normalize to optimal level (-3dB)
        target_peak = 0.707  # -3dB
        if original_peak > 0:
            audio = audio * (target_peak / original_peak)

        # 2. Apply high-pass filter to remove subsonic content
        if sr > 0:
            # Design butterworth high-pass filter
            sos = signal.butter(4, 30, 'hp', fs=sr, output='sos')

            # Apply filter to each channel
            audio_np = audio.cpu().numpy()

            # Ensure contiguous array
            if not audio_np.flags.c_contiguous:
                audio_np = np.ascontiguousarray(audio_np)

            for ch in range(audio_np.shape[0]):
                ch_data = np.ascontiguousarray(audio_np[ch])
                filtered = signal.sosfiltfilt(sos, ch_data)
                audio_np[ch] = filtered.copy()

            audio = torch.from_numpy(audio_np).to(audio.device)

        # 3. Apply gentle limiting to prevent clipping
        audio = torch.tanh(audio * 0.95) / 0.95

        # 4. Add subtle dithering for bit depth preservation
        if audio.dtype == torch.float32:
            dither = torch.randn_like(audio) * 1e-6
            audio = audio + dither

        return audio, original_peak

    def _split_frequency_bands(self, audio: torch.Tensor, sr: int) -> Dict:
        """Split audio into frequency bands for better 'other' stem processing."""
        bands = {}

        # Design filters for different frequency ranges
        # Low-mid: 250-1000 Hz
        # Mid: 1000-4000 Hz (important for 'other' stem)
        # High-mid: 4000-8000 Hz (presence, important for clarity)
        # High: > 8000 Hz (air, brilliance)

        audio_np = audio.cpu().numpy()

        # Ensure contiguous array to avoid negative stride issues
        if not audio_np.flags.c_contiguous:
            audio_np = np.ascontiguousarray(audio_np)

        # Low-mid band (250-1000 Hz) - important for guitars, keys lower register
        sos_lowmid = signal.butter(4, [250, 1000], 'bandpass', fs=sr, output='sos')
        bands['low_mid'] = torch.zeros_like(audio)
        for ch in range(audio_np.shape[0] if audio_np.ndim > 1 else 1):
            if audio_np.ndim > 1:
                ch_data = np.ascontiguousarray(audio_np[ch])
                filtered = signal.sosfiltfilt(sos_lowmid, ch_data)
                bands['low_mid'][ch] = torch.from_numpy(filtered.copy()).to(audio.device)
            else:
                filtered = signal.sosfiltfilt(sos_lowmid, audio_np)
                bands['low_mid'] = torch.from_numpy(filtered.copy()).to(audio.device)
                break

        # Mid band (1000-4000 Hz) - critical for most melodic instruments
        sos_mid = signal.butter(4, [1000, 4000], 'bandpass', fs=sr, output='sos')
        bands['mid'] = torch.zeros_like(audio)
        for ch in range(audio_np.shape[0] if audio_np.ndim > 1 else 1):
            if audio_np.ndim > 1:
                ch_data = np.ascontiguousarray(audio_np[ch])
                filtered = signal.sosfiltfilt(sos_mid, ch_data)
                bands['mid'][ch] = torch.from_numpy(filtered.copy()).to(audio.device)
            else:
                filtered = signal.sosfiltfilt(sos_mid, audio_np)
                bands['mid'] = torch.from_numpy(filtered.copy()).to(audio.device)
                break

        # High-mid band (4000-8000 Hz) - presence and clarity
        sos_highmid = signal.butter(4, [4000, 8000], 'bandpass', fs=sr, output='sos')
        bands['high_mid'] = torch.zeros_like(audio)
        for ch in range(audio_np.shape[0] if audio_np.ndim > 1 else 1):
            if audio_np.ndim > 1:
                ch_data = np.ascontiguousarray(audio_np[ch])
                filtered = signal.sosfiltfilt(sos_highmid, ch_data)
                bands['high_mid'][ch] = torch.from_numpy(filtered.copy()).to(audio.device)
            else:
                filtered = signal.sosfiltfilt(sos_highmid, audio_np)
                bands['high_mid'] = torch.from_numpy(filtered.copy()).to(audio.device)
                break

        # High band (> 8000 Hz) - air and brilliance
        sos_high = signal.butter(4, 8000, 'highpass', fs=sr, output='sos')
        bands['high'] = torch.zeros_like(audio)
        for ch in range(audio_np.shape[0] if audio_np.ndim > 1 else 1):
            if audio_np.ndim > 1:
                ch_data = np.ascontiguousarray(audio_np[ch])
                filtered = signal.sosfiltfilt(sos_high, ch_data)
                bands['high'][ch] = torch.from_numpy(filtered.copy()).to(audio.device)
            else:
                filtered = signal.sosfiltfilt(sos_high, audio_np)
                bands['high'] = torch.from_numpy(filtered.copy()).to(audio.device)
                break

        return bands

    def _run_separation_pass(self, audio: torch.Tensor, sr: int,
                             model_key: str, shifts: int, overlap: float,
                             progress_callback: Optional[Callable] = None,
                             progress_offset: float = 0,
                             progress_scale: float = 1.0) -> Dict:
        """Run a single separation pass with specified model."""
        # Get the model to use
        model = self.models.get(model_key, self.model)

        # Convert audio for model
        model_audio = convert_audio(
            audio.unsqueeze(0), sr,
            model.samplerate,
            model.audio_channels
        )

        # Track progress
        if progress_callback:
            def update_progress(current, total):
                if total > 0:
                    prog = progress_offset + (current / total) * progress_scale * 0.8
                    progress_callback(prog, f"Processing segment {current + 1}/{total}")
        else:
            update_progress = None

        # Run separation
        with torch.no_grad():
            sources = apply_model(
                model, model_audio,
                device=self.device,
                shifts=shifts,
                split=True,
                overlap=overlap,
                progress=update_progress if update_progress else False
            )

        # Extract stems (excluding vocals)
        stems = {}
        for i, name in enumerate(model.sources):
            if name.lower() in self.target_stems:
                stem_audio = sources[0, i]

                # Resample if needed
                if model.samplerate != sr:
                    resampler = torchaudio.transforms.Resample(
                        model.samplerate, sr
                    )
                    stem_audio = resampler(stem_audio)

                stems[name] = stem_audio

        return stems

    def _enhance_other_stem(self, original: torch.Tensor, primary_stems: Dict,
                            secondary_stems: Dict, bands: Dict, sr: int) -> torch.Tensor:
        """Enhanced processing specifically for 'other' stem using multiple techniques."""

        # Start with primary model's 'other' stem
        other_enhanced = primary_stems.get('other', torch.zeros_like(original))

        # If we have secondary model results, blend intelligently
        if 'other' in secondary_stems:
            # Analyze which model performs better in different frequency ranges
            primary_other_fft = torch.fft.rfft(primary_stems['other'].flatten())
            secondary_other_fft = torch.fft.rfft(secondary_stems['other'].flatten())

            # Calculate energy in different frequency ranges
            n_bins = primary_other_fft.shape[0]
            mid_range = (n_bins // 8, n_bins // 2)  # Focus on mid frequencies

            # Use secondary model where it has more energy in mid frequencies
            primary_mid_energy = torch.sum(torch.abs(primary_other_fft[mid_range[0]:mid_range[1]]) ** 2)
            secondary_mid_energy = torch.sum(torch.abs(secondary_other_fft[mid_range[0]:mid_range[1]]) ** 2)

            if secondary_mid_energy > primary_mid_energy * 1.1:  # If secondary is significantly better
                # Blend with emphasis on secondary for mid frequencies
                blend_ratio = 0.6  # 60% secondary, 40% primary
                other_enhanced = primary_stems['other'] * (1 - blend_ratio) + secondary_stems['other'] * blend_ratio

        # Subtract drums and bass more aggressively from critical bands
        residual = original.clone()
        if 'drums' in primary_stems:
            residual = residual - primary_stems['drums'] * 1.1  # Slightly over-subtract drums
        if 'bass' in primary_stems:
            residual = residual - primary_stems['bass'] * 1.05  # Slightly over-subtract bass

        # Focus on mid and high-mid bands where 'other' instruments live
        if 'mid' in bands and 'high_mid' in bands:
            # Extract melodic content from these bands
            melodic_content = bands['mid'] + bands['high_mid'] * 0.8

            # Remove drum transients from melodic content
            melodic_cleaned = self._remove_percussive_transients(melodic_content, sr)

            # Blend with original 'other' stem
            other_enhanced = other_enhanced * 0.7 + melodic_cleaned * 0.3

        # Apply harmonic enhancement specifically for melodic instruments
        other_enhanced = self._enhance_harmonics(other_enhanced, sr)

        return other_enhanced

    def _separate_other_components(self, other_stem: torch.Tensor, sr: int) -> Dict:
        """Separate 'other' stem into sub-components for better quality."""
        components = {}

        # Handle mono/stereo properly
        if other_stem.dim() == 1:
            audio_to_process = other_stem
            original_shape = other_stem.shape
        else:
            # Convert to mono for processing
            audio_to_process = torch.mean(other_stem, dim=0)
            original_shape = other_stem.shape

        # Calculate STFT with proper window
        window = torch.hann_window(4096).to(other_stem.device)
        stft = torch.stft(
            audio_to_process,
            n_fft=4096,
            hop_length=512,
            win_length=4096,
            window=window,
            return_complex=True
        )

        magnitude = torch.abs(stft)
        phase = torch.angle(stft)

        # Median filtering for harmonic/percussive separation
        magnitude_np = magnitude.cpu().numpy()

        # Harmonic: horizontal structures in spectrogram
        harmonic_mag = torch.from_numpy(
            signal.medfilt2d(magnitude_np, kernel_size=(1, 31))
        ).to(magnitude.device)

        # Percussive: vertical structures in spectrogram
        percussive_mag = torch.from_numpy(
            signal.medfilt2d(magnitude_np, kernel_size=(31, 1))
        ).to(magnitude.device)

        # Reconstruct harmonic component
        harmonic_stft = harmonic_mag * torch.exp(1j * phase)
        harmonic_mono = torch.istft(
            harmonic_stft,
            n_fft=4096,
            hop_length=512,
            win_length=4096,
            window=window
        )

        # Reconstruct percussive component
        percussive_stft = percussive_mag * torch.exp(1j * phase)
        percussive_mono = torch.istft(
            percussive_stft,
            n_fft=4096,
            hop_length=512,
            win_length=4096,
            window=window
        )

        # Residual
        residual_mono = audio_to_process[
                        :min(len(audio_to_process), len(harmonic_mono), len(percussive_mono))] - harmonic_mono[:min(
            len(audio_to_process), len(harmonic_mono), len(percussive_mono))] - percussive_mono[
                                                                                :min(len(audio_to_process),
                                                                                     len(harmonic_mono),
                                                                                     len(percussive_mono))]

        # Restore to original shape
        def restore_shape(mono_audio, target_shape):
            if len(target_shape) == 1:
                return mono_audio[:target_shape[0]]
            else:
                # Broadcast to stereo
                restored = mono_audio[:target_shape[1]].unsqueeze(0).expand(target_shape[0], -1)
                return restored[:, :target_shape[1]]

        components['harmonic'] = restore_shape(harmonic_mono, original_shape)
        components['percussive'] = restore_shape(percussive_mono, original_shape)
        components['residual'] = restore_shape(residual_mono, original_shape)

        # Further separate harmonic into pitched and unpitched
        components['pitched'] = self._extract_pitched_content(components['harmonic'], sr)
        components['unpitched'] = components['harmonic'] - components['pitched']

        return components

    def _remove_percussive_transients(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Remove percussive transients from audio to clean up 'other' stem."""
        # Handle mono/stereo
        if audio.dim() == 1:
            audio_to_process = audio
            original_shape = audio.shape
        else:
            audio_to_process = torch.mean(audio, dim=0)
            original_shape = audio.shape

        # Detect transients with proper window
        window = torch.hann_window(2048).to(audio.device)
        stft = torch.stft(
            audio_to_process,
            n_fft=2048,
            hop_length=256,
            win_length=2048,
            window=window,
            return_complex=True
        )

        magnitude = torch.abs(stft)

        # Calculate spectral flux
        flux = torch.diff(magnitude, dim=1)
        flux = torch.relu(flux)
        flux_sum = torch.sum(flux, dim=0)

        # Find transient peaks
        flux_mean = torch.mean(flux_sum)
        flux_std = torch.std(flux_sum)
        transient_threshold = flux_mean + 2 * flux_std

        # Create mask
        transient_mask = flux_sum > transient_threshold
        suppression_mask = torch.ones_like(flux_sum)
        suppression_mask[transient_mask] = 0.3

        # Pad mask
        suppression_mask = torch.nn.functional.pad(suppression_mask.unsqueeze(0), (1, 0))

        # Apply mask
        magnitude_cleaned = magnitude * suppression_mask

        # Reconstruct
        phase = torch.angle(stft)
        stft_cleaned = magnitude_cleaned * torch.exp(1j * phase)
        audio_cleaned_mono = torch.istft(
            stft_cleaned,
            n_fft=2048,
            hop_length=256,
            win_length=2048,
            window=window
        )

        # Restore shape
        if len(original_shape) == 1:
            return audio_cleaned_mono[:original_shape[0]]
        else:
            restored = audio_cleaned_mono[:original_shape[1]].unsqueeze(0).expand(original_shape[0], -1)
            return restored[:, :original_shape[1]]

    def _spectral_cleaning(self, audio: torch.Tensor, sr: int, stem_type: str) -> torch.Tensor:
        """Clean spectrum based on stem type."""
        # Handle shape properly
        if audio.dim() == 1:
            audio_to_process = audio
            original_shape = audio.shape
        else:
            audio_to_process = torch.mean(audio, dim=0)
            original_shape = audio.shape

        # Convert to frequency domain with proper window
        window = torch.hann_window(2048).to(audio.device)
        stft = torch.stft(
            audio_to_process,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window=window,
            return_complex=True
        )

        magnitude = torch.abs(stft)
        phase = torch.angle(stft)

        # Apply stem-specific filtering
        if stem_type == "drums":
            magnitude = self._enhance_transients_spectral(magnitude)
        elif stem_type == "bass":
            magnitude = self._enhance_bass_spectral(magnitude, sr)
        else:  # "other"
            magnitude = self._balanced_spectral_cleaning(magnitude)

        # Reconstruct
        stft_cleaned = magnitude * torch.exp(1j * phase)
        audio_cleaned_mono = torch.istft(
            stft_cleaned,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            window=window
        )

        # Restore shape
        if len(original_shape) == 1:
            return audio_cleaned_mono[:original_shape[0]]
        else:
            restored = audio_cleaned_mono[:original_shape[1]].unsqueeze(0).expand(original_shape[0], -1)
            return restored[:, :original_shape[1]]

    def _advanced_spectral_cleaning(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Advanced spectral cleaning using multiple techniques."""
        # Handle shape properly
        if audio.dim() == 1:
            audio_to_process = audio
            original_shape = audio.shape
        else:
            audio_to_process = torch.mean(audio, dim=0)
            original_shape = audio.shape

        # Large FFT for precise frequency work with proper window
        window = torch.hann_window(4096).to(audio.device)
        stft = torch.stft(
            audio_to_process,
            n_fft=4096,
            hop_length=256,
            win_length=4096,
            window=window,
            return_complex=True
        )

        magnitude = torch.abs(stft)
        phase = torch.angle(stft)

        # Spectral subtraction for noise reduction
        noise_profile = torch.min(magnitude, dim=1)[0].unsqueeze(1)
        magnitude_denoised = torch.relu(magnitude - noise_profile * 0.5)

        # Wiener filtering
        signal_power = magnitude_denoised ** 2
        noise_power = noise_profile ** 2
        wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
        magnitude_filtered = magnitude_denoised * wiener_gain

        # Spectral smoothing
        magnitude_np = magnitude_filtered.cpu().numpy()
        magnitude_smooth = gaussian_filter1d(magnitude_np, sigma=2, axis=0)
        magnitude_smooth = gaussian_filter1d(magnitude_smooth, sigma=1, axis=1)
        magnitude_final = torch.from_numpy(magnitude_smooth).to(magnitude.device)

        # Reconstruct
        stft_cleaned = magnitude_final * torch.exp(1j * phase)
        audio_cleaned_mono = torch.istft(
            stft_cleaned,
            n_fft=4096,
            hop_length=256,
            win_length=4096,
            window=window
        )

        # Restore shape
        if len(original_shape) == 1:
            return audio_cleaned_mono[:original_shape[0]]
        else:
            restored = audio_cleaned_mono[:original_shape[1]].unsqueeze(0).expand(original_shape[0], -1)
            return restored[:, :original_shape[1]]


    def _enhance_harmonics(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Enhance harmonic content for better melodic instrument clarity."""
        # FFT
        fft = torch.fft.rfft(audio.flatten())
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)

        # Find peaks (likely harmonics)
        mag_np = magnitude.cpu().numpy()
        peaks, properties = signal.find_peaks(
            mag_np,
            height=np.max(mag_np) * 0.1,  # At least 10% of max
            distance=20  # Minimum distance between peaks
        )

        # Create enhancement curve
        enhancement = torch.ones_like(magnitude)
        peak_indices = torch.tensor(peaks).to(magnitude.device)

        # Enhance around peaks (harmonics)
        for peak in peak_indices:
            if peak < len(enhancement):
                # Enhance peak and nearby bins
                start = max(0, peak - 5)
                end = min(len(enhancement), peak + 6)
                enhancement[start:end] *= 1.3  # 30% boost

        # Apply enhancement
        magnitude_enhanced = magnitude * enhancement

        # Reconstruct
        fft_enhanced = magnitude_enhanced * torch.exp(1j * phase)
        audio_enhanced = torch.fft.irfft(fft_enhanced, n=audio.flatten().shape[0])

        # Reshape
        if audio.dim() > 1:
            audio_enhanced = audio_enhanced.reshape(audio.shape)

        return audio_enhanced

    def _extract_pitched_content(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract pitched/tonal content from audio."""
        # Use autocorrelation to enhance pitched content
        audio_flat = audio.flatten()

        # Compute autocorrelation
        autocorr = torch.nn.functional.conv1d(
            audio_flat.unsqueeze(0).unsqueeze(0),
            audio_flat.flip(0).unsqueeze(0).unsqueeze(0),
            padding=len(audio_flat)
        ).squeeze()

        # Find periodicity (pitched content has strong periodicity)
        autocorr_normalized = autocorr / (torch.max(torch.abs(autocorr)) + 1e-10)

        # Use comb filter based on detected periodicity
        # This enhances harmonic content
        delay_samples = 512  # Roughly 11ms at 44.1kHz
        comb_filtered = audio_flat.clone()

        if len(audio_flat) > delay_samples:
            delayed = torch.nn.functional.pad(audio_flat, (delay_samples, 0))[:-delay_samples]
            comb_filtered = audio_flat * 0.7 + delayed * 0.3

        # Reshape
        if audio.dim() > 1:
            comb_filtered = comb_filtered.reshape(audio.shape)

        return comb_filtered

    def _refine_separation_focused(self, original: torch.Tensor, stems: Dict,
                                   other_components: Dict, sr: int) -> Dict:
        """Refined separation with focus on 'other' stem quality."""
        refined_stems = {}

        # Keep drums and bass as-is (they're already excellent)
        refined_stems['drums'] = stems.get('drums', torch.zeros_like(original))
        refined_stems['bass'] = stems.get('bass', torch.zeros_like(original))

        # Refine 'other' stem using components
        if 'other' in stems and other_components:
            # Combine components intelligently
            # Prioritize harmonic (melodic) content
            other_refined = other_components['harmonic'] * 0.6

            # Add some pitched content for clarity
            if 'pitched' in other_components:
                other_refined += other_components['pitched'] * 0.3

            # Add minimal percussive elements (might be percussion instruments)
            if 'percussive' in other_components:
                other_refined += other_components['percussive'] * 0.1

            # Ensure energy conservation
            original_energy = torch.mean(stems['other'] ** 2)
            refined_energy = torch.mean(other_refined ** 2)
            if refined_energy > 0:
                energy_scale = torch.sqrt(original_energy / refined_energy)
                other_refined *= energy_scale

            refined_stems['other'] = other_refined
        else:
            refined_stems['other'] = stems.get('other', torch.zeros_like(original))

        # Create additional detailed stems from 'other'
        if self.config.get('create_sub_stems', True):
            # Piano/Keys (focus on mid frequencies with harmonic content)
            refined_stems['keys'] = self._extract_keys(refined_stems['other'], sr)

            # Guitar (focus on mid-high with both harmonic and percussive)
            refined_stems['guitar'] = self._extract_guitar(refined_stems['other'], sr)

            # Synth/Pads (sustained harmonic content)
            refined_stems['synth'] = self._extract_synth(refined_stems['other'], sr)

            # Update 'other' to be residual after extracting sub-stems
            refined_stems['other_misc'] = refined_stems['other'] - (
                    refined_stems['keys'] + refined_stems['guitar'] + refined_stems['synth']
            )

        return refined_stems

    def _extract_keys(self, other_stem: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract piano/keyboard-like content from 'other' stem."""
        # Focus on 200-2000 Hz range typical for piano
        sos = signal.butter(4, [200, 2000], 'bandpass', fs=sr, output='sos')

        keys = torch.zeros_like(other_stem)
        audio_np = other_stem.cpu().numpy()

        # Ensure contiguous array
        if not audio_np.flags.c_contiguous:
            audio_np = np.ascontiguousarray(audio_np)

        for ch in range(audio_np.shape[0] if audio_np.ndim > 1 else 1):
            if audio_np.ndim > 1:
                ch_data = np.ascontiguousarray(audio_np[ch])
                filtered = signal.sosfiltfilt(sos, ch_data).copy()
            else:
                filtered = signal.sosfiltfilt(sos, audio_np).copy()

            # Detect note onsets typical of keys
            onset_env = np.abs(filtered) + np.abs(np.diff(filtered, prepend=0))
            onset_smooth = gaussian_filter1d(onset_env, sigma=50)

            # Apply envelope to emphasize note attacks
            if audio_np.ndim > 1:
                keys[ch] = torch.from_numpy(filtered * (1 + onset_smooth * 0.5)).to(other_stem.device)
            else:
                keys = torch.from_numpy(filtered * (1 + onset_smooth * 0.5)).to(other_stem.device)
                break

        return keys * 0.3  # Scale down to avoid over-emphasis

    def _extract_guitar(self, other_stem: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract guitar-like content from 'other' stem."""
        # Guitar fundamental range: 80-1200 Hz with rich harmonics up to 5kHz
        sos_fundamental = signal.butter(4, [80, 1200], 'bandpass', fs=sr, output='sos')
        sos_harmonics = signal.butter(4, [1200, 5000], 'bandpass', fs=sr, output='sos')

        guitar = torch.zeros_like(other_stem)
        audio_np = other_stem.cpu().numpy()

        # Ensure contiguous array
        if not audio_np.flags.c_contiguous:
            audio_np = np.ascontiguousarray(audio_np)

        for ch in range(audio_np.shape[0] if audio_np.ndim > 1 else 1):
            if audio_np.ndim > 1:
                ch_data = np.ascontiguousarray(audio_np[ch])
                fundamental = signal.sosfiltfilt(sos_fundamental, ch_data).copy()
                harmonics = signal.sosfiltfilt(sos_harmonics, ch_data).copy()
            else:
                fundamental = signal.sosfiltfilt(sos_fundamental, audio_np).copy()
                harmonics = signal.sosfiltfilt(sos_harmonics, audio_np).copy()

            # Combine with emphasis on fundamental
            combined = fundamental * 0.7 + harmonics * 0.3

            # Add slight distortion characteristic of guitar
            combined = np.tanh(combined * 1.5) / 1.5

            if audio_np.ndim > 1:
                guitar[ch] = torch.from_numpy(combined).to(other_stem.device)
            else:
                guitar = torch.from_numpy(combined).to(other_stem.device)
                break

        return guitar * 0.3

    def _extract_synth(self, other_stem: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract synthesizer/pad content from 'other' stem."""
        # Synths often have sustained energy and rich harmonics
        synth = torch.zeros_like(other_stem)

        # Calculate sustained energy regions
        window_size = int(sr * 0.1)  # 100ms windows
        hop = window_size // 2

        audio_np = other_stem.cpu().numpy()

        for ch in range(audio_np.shape[0] if audio_np.ndim > 1 else 1):
            if audio_np.ndim > 1:
                signal_ch = audio_np[ch]
            else:
                signal_ch = audio_np

            # Calculate energy envelope
            energy_env = []
            for i in range(0, len(signal_ch) - window_size, hop):
                window = signal_ch[i:i + window_size]
                energy_env.append(np.mean(window ** 2))

            energy_env = np.array(energy_env)

            # Find sustained regions (low variance in energy)
            if len(energy_env) > 10:
                energy_var = np.convolve(energy_env, np.ones(10) / 10, mode='same')
                sustained_mask = energy_var > np.median(energy_var)

                # Interpolate mask to full size
                sustained_full = np.interp(
                    np.arange(len(signal_ch)),
                    np.arange(len(sustained_mask)) * hop,
                    sustained_mask.astype(float)
                )

                # Apply high-pass filter to get synth brightness
                sos_bright = signal.butter(2, 500, 'highpass', fs=sr, output='sos')
                bright = signal.sosfiltfilt(sos_bright, signal_ch)

                # Combine sustained regions with brightness
                synth_ch = signal_ch * sustained_full * 0.5 + bright * sustained_full * 0.5
            else:
                synth_ch = signal_ch * 0.3

            if audio_np.ndim > 1:
                synth[ch] = torch.from_numpy(synth_ch).to(other_stem.device)
            else:
                synth = torch.from_numpy(synth_ch).to(other_stem.device)
                break

        return synth * 0.3

    def _postprocess_stems_enhanced(self, stems: Dict, original: torch.Tensor,
                                    sr: int, original_peak: float) -> Dict:
        """Enhanced post-processing with focus on 'other' stem quality."""
        processed_stems = {}

        for name, stem in stems.items():
            if name in ['drums', 'bass']:
                # Light processing for drums and bass (already good)
                stem_processed = self._adaptive_gate(stem, threshold_db=-45)
            elif name == 'other' or 'other' in name:
                # Heavy processing for 'other' and its sub-stems
                # 1. Phase alignment
                stem_aligned = self._align_phase(stem, original)

                # 2. Advanced spectral cleaning
                stem_cleaned = self._advanced_spectral_cleaning(stem_aligned, sr)

                # 3. Harmonic enhancement
                stem_enhanced = self._enhance_harmonics(stem_cleaned, sr)

                # 4. Dynamic EQ based on content
                stem_eq = self._dynamic_eq(stem_enhanced, sr, name)

                # 5. Multiband compression for consistency
                stem_compressed = self._multiband_compress(stem_eq, sr)

                # 6. Final gating
                stem_processed = self._adaptive_gate(stem_compressed, threshold_db=-50)
            else:
                # Moderate processing for sub-stems
                stem_aligned = self._align_phase(stem, original)
                stem_cleaned = self._spectral_cleaning(stem_aligned, sr, stem_type=name)
                stem_processed = self._adaptive_gate(stem_cleaned, threshold_db=-48)

            # Restore levels
            if original_peak > 0:
                current_peak = torch.max(torch.abs(stem_processed)).item()
                if current_peak > 0:
                    stem_energy = torch.mean(stem_processed ** 2)
                    original_energy = torch.mean(original ** 2)
                    energy_ratio = torch.sqrt(stem_energy / (original_energy + 1e-10))
                    stem_processed = stem_processed * (original_peak / 0.707) * energy_ratio

            processed_stems[name] = stem_processed

        return processed_stems

    def _align_phase(self, stem: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Align phase of stem with reference signal."""
        if stem.shape != reference.shape:
            return stem

        # Use cross-correlation to find best alignment
        for ch in range(stem.shape[0]):
            correlation = torch.nn.functional.conv1d(
                reference[ch:ch + 1].unsqueeze(0),
                stem[ch:ch + 1].flip(-1).unsqueeze(0),
                padding=100
            )

            # Find peak correlation
            peak_idx = torch.argmax(torch.abs(correlation))
            shift = peak_idx - 100

            # Apply shift if significant
            if abs(shift) > 0 and abs(shift) < 50:
                stem[ch] = torch.roll(stem[ch], shifts=int(shift))

        return stem



    def _enhance_transients_spectral(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Enhance transients in spectral domain."""
        # Detect transients using spectral flux
        flux = torch.diff(magnitude, dim=1)
        flux_positive = torch.relu(flux)

        # Create transient mask
        transient_mask = flux_positive / (torch.max(flux_positive) + 1e-10)
        transient_mask = torch.nn.functional.pad(transient_mask, (1, 0))

        # Enhance transients
        enhanced = magnitude * (1 + transient_mask * 0.5)

        return enhanced

    def _enhance_bass_spectral(self, magnitude: torch.Tensor, sr: int) -> torch.Tensor:
        """Enhance bass frequencies."""
        n_bins = magnitude.shape[0]
        freq_bins = torch.arange(n_bins) * sr / (2 * n_bins)

        # Create bass emphasis curve
        bass_curve = torch.exp(-freq_bins / 500).to(magnitude.device)
        bass_curve = bass_curve.unsqueeze(1)

        # Apply emphasis
        enhanced = magnitude * (0.7 + 0.3 * bass_curve)

        return enhanced

    def _balanced_spectral_cleaning(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Balanced spectral cleaning for 'other' stems."""
        # Apply gentle smoothing
        magnitude_np = magnitude.cpu().numpy()
        smoothed = gaussian_filter1d(magnitude_np, sigma=1.5, axis=1)

        return torch.from_numpy(smoothed).to(magnitude.device)



    def _dynamic_eq(self, audio: torch.Tensor, sr: int, stem_type: str) -> torch.Tensor:
        """Apply dynamic EQ based on stem type."""
        # Define EQ curves for different stem types
        eq_curves = {
            'other': [(200, -2), (500, 0), (1000, 2), (2000, 3), (4000, 2), (8000, 1)],
            'keys': [(200, 0), (500, 2), (1000, 3), (2000, 2), (4000, 0)],
            'guitar': [(100, -1), (250, 2), (500, 3), (1000, 2), (2500, 3), (5000, 1)],
            'synth': [(100, 1), (500, 0), (1000, 1), (4000, 2), (8000, 3)],
            'other_misc': [(500, 0), (1000, 1), (2000, 1), (4000, 0)]
        }

        curve = eq_curves.get(stem_type, eq_curves['other'])

        # Apply EQ in frequency domain
        fft = torch.fft.rfft(audio.flatten())
        freqs = torch.fft.rfftfreq(audio.flatten().shape[0], 1 / sr)

        # Create EQ curve
        eq_gain = torch.ones_like(freqs)
        for freq, gain_db in curve:
            mask = torch.abs(freqs - freq) < freq * 0.2  # ±20% bandwidth
            eq_gain[mask] *= 10 ** (gain_db / 20)

        # Smooth the EQ curve
        eq_gain_np = gaussian_filter1d(eq_gain.cpu().numpy(), sigma=10)
        eq_gain = torch.from_numpy(eq_gain_np).to(fft.device)

        # Apply EQ
        fft_eq = fft * eq_gain
        audio_eq = torch.fft.irfft(fft_eq, n=audio.flatten().shape[0])

        if audio.dim() > 1:
            audio_eq = audio_eq.reshape(audio.shape)

        return audio_eq

    def _multiband_compress(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply multiband compression for consistent dynamics."""
        # Split into 4 bands
        bands = []
        band_limits = [(0, 200), (200, 800), (800, 3000), (3000, sr // 2)]

        audio_np = audio.cpu().numpy()

        # Ensure contiguous array
        if not audio_np.flags.c_contiguous:
            audio_np = np.ascontiguousarray(audio_np)

        compressed_bands = []

        for low, high in band_limits:
            # Filter band
            if low == 0:
                sos = signal.butter(4, high, 'lowpass', fs=sr, output='sos')
            elif high == sr // 2:
                sos = signal.butter(4, low, 'highpass', fs=sr, output='sos')
            else:
                sos = signal.butter(4, [low, high], 'bandpass', fs=sr, output='sos')

            band_data = np.ascontiguousarray(audio_np.flatten())
            band = signal.sosfiltfilt(sos, band_data).copy()

            # Simple compression
            threshold = np.percentile(np.abs(band), 85)
            ratio = 3.0

            compressed = np.where(
                np.abs(band) > threshold,
                np.sign(band) * (threshold + (np.abs(band) - threshold) / ratio),
                band
            )

            compressed_bands.append(compressed)

        # Sum compressed bands
        audio_compressed = np.sum(compressed_bands, axis=0)

        # Convert back to tensor and reshape
        audio_compressed = torch.from_numpy(audio_compressed).to(audio.device)
        if audio.dim() > 1:
            audio_compressed = audio_compressed.reshape(audio.shape)

        return audio_compressed


    def _adaptive_gate(self, audio: torch.Tensor, threshold_db: float = -50) -> torch.Tensor:
        """Apply adaptive noise gate."""
        # Convert threshold to linear
        threshold_linear = 10 ** (threshold_db / 20)

        # Calculate envelope
        envelope = self._get_envelope(audio, 512, 256)

        # Create gate curve
        gate_curve = torch.sigmoid((envelope - threshold_linear) * 100)

        # Interpolate to full size
        gate_curve_full = torch.nn.functional.interpolate(
            gate_curve.unsqueeze(0),
            size=audio.shape[-1],
            mode='linear'
        ).squeeze()

        # Apply gate
        if audio.dim() == 1:
            return audio * gate_curve_full
        else:
            return audio * gate_curve_full.unsqueeze(0)

    def _get_envelope(self, audio: torch.Tensor, window_size: int,
                      hop: int) -> torch.Tensor:
        """Get energy envelope of audio."""
        # Ensure audio is 2D
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Calculate energy in windows
        n_frames = (audio.shape[-1] - window_size) // hop + 1
        envelope = torch.zeros(audio.shape[0], n_frames)

        for i in range(n_frames):
            start = i * hop
            end = start + window_size
            envelope[:, i] = torch.mean(audio[:, start:end] ** 2, dim=-1)

        return torch.sqrt(envelope + 1e-10)

    def _calculate_enhanced_metrics(self, original: torch.Tensor,
                                    stems: Dict) -> Dict:
        """Calculate comprehensive quality metrics."""
        metrics = {}

        # Reconstruct from stems
        reconstructed = torch.zeros_like(original)
        for stem in stems.values():
            if stem.shape == original.shape:
                reconstructed += stem

        # Signal quality metrics
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean((original - reconstructed) ** 2)
        metrics["snr_db"] = float(10 * torch.log10(signal_power / (noise_power + 1e-10)))

        # Correlation
        if original.numel() > 0:
            correlation = torch.corrcoef(torch.stack([
                original.flatten(),
                reconstructed.flatten()
            ]))[0, 1]
            metrics["correlation"] = float(correlation) if not torch.isnan(correlation) else 0.0

        # Per-stem metrics
        total_energy = torch.sum(original ** 2)
        for name, stem in stems.items():
            stem_energy = torch.sum(stem ** 2)
            metrics[f"{name}_energy_ratio"] = float(stem_energy / (total_energy + 1e-10))

            # Calculate stem SNR
            stem_noise = original - stem if stem.shape == original.shape else stem
            stem_signal_power = torch.mean(stem ** 2)
            stem_noise_power = torch.mean(stem_noise ** 2)
            metrics[f"{name}_snr_db"] = float(
                10 * torch.log10(stem_signal_power / (stem_noise_power + 1e-10))
            )

        # Overall quality score
        metrics["quality_score"] = min(
            (metrics["snr_db"] / 30) * 50 +
            metrics.get("correlation", 0) * 50,
            100.0
        )

        metrics["processing_mode"] = "enhanced_two_pass"
        metrics["num_stems"] = len(stems)

        return metrics