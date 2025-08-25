#!/usr/bin/env python3
"""
DDSP-Style Audio Enhancement - Simplified Runnable Version
Enhance audio using signal processing for timbre modification.
No external DDSP dependencies required.
"""

import time
import torch
import torchaudio
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy import signal
from scipy.signal import butter, sosfilt, hilbert
import warnings

warnings.filterwarnings('ignore')

console = Console()


class SimpleDDSPEnhancer:
    """
    Simplified DDSP-style audio enhancer for timbre transfer.
    Uses signal processing techniques to modify timbre without ML models.
    """

    def __init__(self):
        self.sample_rate = 44100  # Standard rate
        self.device = "cpu"
        self.frame_size = 2048
        self.hop_size = 512

        # Timbre profiles for different instruments
        self.timbre_profiles = {
            "violin": {
                "harmonics": [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2],
                "formants": [700, 1400, 2800, 4000],  # Violin formant frequencies
                "vibrato_rate": 5.5,  # Hz
                "vibrato_depth": 0.02,
                "brightness": 1.3,
                "warmth": 0.8
            },
            "flute": {
                "harmonics": [1.0, 0.3, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01],
                "formants": [800, 1800, 3000],
                "vibrato_rate": 4.5,
                "vibrato_depth": 0.015,
                "brightness": 1.1,
                "warmth": 0.6
            },
            "cello": {
                "harmonics": [1.0, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35],
                "formants": [400, 1000, 2000, 3200],
                "vibrato_rate": 4.0,
                "vibrato_depth": 0.025,
                "brightness": 0.8,
                "warmth": 1.2
            },
            "piano": {
                "harmonics": [1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1],
                "formants": [600, 1200, 2400, 3600],
                "vibrato_rate": 0,  # No vibrato for piano
                "vibrato_depth": 0,
                "brightness": 1.0,
                "warmth": 1.0
            }
        }

        self.current_profile = None
        self.enhancement_level = 0.5

    def load_model(self, model_type="violin"):
        """Load timbre profile for the target instrument."""
        console.print(Panel.fit(
            "[bold cyan]Loading Timbre Profile[/bold cyan]\n"
            f"Target Instrument: [yellow]{model_type}[/yellow]\n"
            "[dim]Preparing signal processing parameters[/dim]",
            title="🎵 Model Loading"
        ))

        if model_type not in self.timbre_profiles:
            console.print(f"[yellow]Warning: Unknown instrument '{model_type}', using violin[/yellow]")
            model_type = "violin"

        self.current_profile = self.timbre_profiles[model_type]
        self.model_type = model_type

        console.print(f"[green]✓[/green] Timbre profile ready for {model_type}\n")

    def configure(self, enhancement_level=0.5):
        """Configure enhancement settings."""
        console.print("[bold]Configuring enhancement settings...[/bold]")

        self.enhancement_level = np.clip(enhancement_level, 0.0, 1.0)

        # Display settings
        table = Table(title="Enhancement Settings")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_row("Sample Rate", f"{self.sample_rate} Hz")
        table.add_row("Enhancement Level", f"{self.enhancement_level:.1%}")
        table.add_row("Target Timbre", self.model_type.title())
        table.add_row("Processing Mode", "Spectral Timbre Transfer")

        if self.current_profile:
            table.add_row("Vibrato", f"{self.current_profile['vibrato_rate']} Hz"
            if self.current_profile['vibrato_rate'] > 0 else "None")
            table.add_row("Brightness", f"{self.current_profile['brightness']:.1f}x")

        console.print(table)
        console.print("[green]✓[/green] Settings applied\n")

    def extract_features(self, audio):
        """Extract audio features using signal processing."""
        console.print("Extracting audio features...")

        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio_np = audio.numpy().squeeze()
        else:
            audio_np = audio.squeeze() if audio.ndim > 1 else audio

        # Extract pitch using autocorrelation
        f0_sequence = self._extract_pitch_autocorr(audio_np)

        # Extract loudness envelope
        loudness = self._extract_loudness(audio_np)

        # Extract spectral envelope
        spectral_envelope = self._extract_spectral_envelope(audio_np)

        console.print(f"[green]✓[/green] Features extracted: {len(f0_sequence)} pitch frames\n")

        return {
            'f0': f0_sequence,
            'loudness': loudness,
            'spectral_envelope': spectral_envelope,
            'audio': audio_np
        }

    def _extract_pitch_autocorr(self, audio):
        """Extract fundamental frequency using autocorrelation."""
        frame_length = 2048
        hop_length = 512

        pitches = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]

            # Apply window
            frame = frame * np.hanning(len(frame))

            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            # Find peaks
            min_period = int(self.sample_rate / 800)  # 800 Hz max
            max_period = int(self.sample_rate / 80)  # 80 Hz min

            if max_period < len(autocorr):
                autocorr_slice = autocorr[min_period:max_period]
                if len(autocorr_slice) > 0 and np.max(autocorr_slice) > 0:
                    peak_idx = np.argmax(autocorr_slice) + min_period
                    f0 = self.sample_rate / peak_idx
                else:
                    f0 = 0
            else:
                f0 = 0

            pitches.append(f0)

        return np.array(pitches)

    def _extract_loudness(self, audio):
        """Extract loudness envelope."""
        frame_length = 2048
        hop_length = 512

        loudness = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            db = 20 * np.log10(rms + 1e-10)
            loudness.append(db)

        return np.array(loudness)

    def _extract_spectral_envelope(self, audio):
        """Extract spectral envelope using cepstral analysis."""
        frame_length = 2048
        hop_length = 512

        envelopes = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]

            # Apply window
            frame = frame * np.hanning(len(frame))

            # FFT
            spectrum = np.fft.rfft(frame)
            magnitude = np.abs(spectrum)

            # Cepstral smoothing for envelope
            log_magnitude = np.log(magnitude + 1e-10)
            cepstrum = np.fft.irfft(log_magnitude)

            # Lifter to extract envelope
            lifter_cutoff = 30
            cepstrum[lifter_cutoff:-lifter_cutoff] = 0

            envelope = np.exp(np.fft.rfft(cepstrum).real)
            envelopes.append(envelope)

        return envelopes

    def enhance_audio(self, features):
        """Apply timbre enhancement using signal processing."""
        console.print("Applying timbre enhancement...")

        audio = features['audio']
        f0_sequence = features['f0']
        loudness = features['loudness']

        # Step 1: Harmonic modification
        enhanced = self._modify_harmonics(audio, f0_sequence)

        # Step 2: Apply formant filtering
        enhanced = self._apply_formants(enhanced)

        # Step 3: Add vibrato if applicable
        if self.current_profile['vibrato_rate'] > 0:
            enhanced = self._add_vibrato(enhanced, f0_sequence)

        # Step 4: Adjust brightness and warmth
        enhanced = self._adjust_spectral_balance(enhanced)

        # Step 5: Mix with original based on enhancement level
        enhanced = (1 - self.enhancement_level) * audio + self.enhancement_level * enhanced

        # Normalize
        max_val = np.max(np.abs(enhanced))
        if max_val > 0.95:
            enhanced = enhanced * (0.95 / max_val)

        console.print(f"[green]✓[/green] Enhancement applied with {self.model_type} timbre\n")

        return enhanced

    def _modify_harmonics(self, audio, f0_sequence):
        """Modify harmonic structure to match target instrument."""
        # Use STFT for harmonic modification
        nperseg = 2048
        noverlap = nperseg // 2

        f, t, Zxx = signal.stft(audio, fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap)

        # Modify harmonics based on profile
        harmonics = self.current_profile['harmonics']

        # Simple harmonic emphasis
        for i, amp in enumerate(harmonics[:4]):  # Process first 4 harmonics
            if i == 0:
                continue
            # Enhance specific harmonic bands
            harmonic_freq = (i + 1) * 220  # Based on A3
            freq_bin = int(harmonic_freq * nperseg / self.sample_rate)
            if freq_bin < len(f):
                bandwidth = 3
                start = max(0, freq_bin - bandwidth)
                end = min(len(f), freq_bin + bandwidth)
                Zxx[start:end] *= (1 + (amp - 0.5) * 0.5)

        # Reconstruct
        _, enhanced = signal.istft(Zxx, fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap)

        # Ensure same length
        if len(enhanced) > len(audio):
            enhanced = enhanced[:len(audio)]
        else:
            enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))

        return enhanced

    def _apply_formants(self, audio):
        """Apply formant filtering to shape timbre."""
        formants = self.current_profile['formants']
        filtered = audio.copy()

        for formant_freq in formants:
            # Create resonant filter for each formant
            Q = 5  # Quality factor
            b, a = signal.iirpeak(formant_freq, Q, self.sample_rate)

            # Apply with reduced gain to avoid overdrive
            formant_signal = signal.filtfilt(b, a, audio) * 0.3
            filtered += formant_signal

        # Normalize
        filtered = filtered / (len(formants) * 0.3 + 1)

        return filtered

    def _add_vibrato(self, audio, f0_sequence):
        """Add vibrato effect."""
        vibrato_rate = self.current_profile['vibrato_rate']
        vibrato_depth = self.current_profile['vibrato_depth']

        if vibrato_rate == 0:
            return audio

        # Generate vibrato LFO
        t = np.arange(len(audio)) / self.sample_rate
        lfo = np.sin(2 * np.pi * vibrato_rate * t) * vibrato_depth

        # Apply pitch modulation using variable delay
        delay_samples = (1 + lfo) * 10  # Small delay modulation

        # Simple implementation using interpolation
        indices = np.arange(len(audio))
        shifted_indices = indices - delay_samples
        shifted_indices = np.clip(shifted_indices, 0, len(audio) - 1)

        # Interpolate
        enhanced = np.interp(indices, shifted_indices, audio)

        return enhanced

    def _adjust_spectral_balance(self, audio):
        """Adjust brightness and warmth of the audio."""
        brightness = self.current_profile['brightness']
        warmth = self.current_profile['warmth']

        # Brightness adjustment (high frequency emphasis)
        if brightness != 1.0:
            # High shelf filter
            freq_cutoff = 4000
            b, a = signal.butter(2, freq_cutoff / (self.sample_rate / 2), 'high')
            high_freq = signal.filtfilt(b, a, audio)
            audio = audio + (brightness - 1.0) * high_freq * 0.5

        # Warmth adjustment (low-mid frequency emphasis)
        if warmth != 1.0:
            # Low shelf filter
            freq_cutoff = 500
            b, a = signal.butter(2, freq_cutoff / (self.sample_rate / 2), 'low')
            low_freq = signal.filtfilt(b, a, audio)
            audio = audio + (warmth - 1.0) * low_freq * 0.3

        return audio

    def process(self, input_file, output_file):
        """Process audio file with timbre enhancement."""

        # Step 1: Load audio
        console.print(f"[bold]Loading audio file:[/bold] {Path(input_file).name}")

        try:
            audio, original_sr = torchaudio.load(input_file)
        except Exception as e:
            console.print(f"[red]Error loading audio: {e}[/red]")
            raise

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            console.print("Converting stereo to mono...")
            audio = audio.mean(dim=0, keepdim=True)

        # Resample if needed
        if original_sr != self.sample_rate:
            console.print(f"Resampling from {original_sr} Hz to {self.sample_rate} Hz...")
            resampler = torchaudio.transforms.Resample(original_sr, self.sample_rate)
            audio = resampler(audio)

        duration = audio.shape[1] / self.sample_rate
        console.print(f"[green]✓[/green] Audio ready: {duration:.1f} seconds\n")

        # Step 2: Extract features
        console.print(Panel.fit(
            "[bold magenta]Timbre Processing[/bold magenta]\n"
            f"Target: {self.model_type.title()} sound",
            title="🎨 Processing"
        ))

        features = self.extract_features(audio)

        # Step 3: Apply enhancement
        start_time = time.time()
        enhanced_audio = self.enhance_audio(features)
        elapsed = time.time() - start_time

        console.print(f"Processing completed in {elapsed:.1f} seconds")

        # Step 4: Save output
        console.print("Saving enhanced audio...")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert back to tensor
        enhanced_tensor = torch.from_numpy(enhanced_audio).float().unsqueeze(0)

        # Resample back to original if needed
        if original_sr != self.sample_rate:
            console.print(f"Resampling back to {original_sr} Hz...")
            resampler = torchaudio.transforms.Resample(self.sample_rate, original_sr)
            enhanced_tensor = resampler(enhanced_tensor)

        torchaudio.save(str(output_path), enhanced_tensor, original_sr)

        # Show results
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        console.print(f"\n[bold green]✨ Enhancement Complete![/bold green]")
        console.print(f"Output: {output_path.name} ({file_size_mb:.2f} MB)")
        console.print(f"Target Timbre: {self.model_type.title()}")
        console.print(f"Enhancement Level: {self.enhancement_level:.0%}")
        console.print(f"Location: {output_path.absolute()}\n")

        # Audio characteristics summary
        table = Table(title="Audio Characteristics")
        table.add_column("Property", style="cyan")
        table.add_column("Original", style="yellow")
        table.add_column("Enhanced", style="green")

        # Calculate some basic stats
        orig_rms = np.sqrt(np.mean(features['audio'] ** 2))
        enh_rms = np.sqrt(np.mean(enhanced_audio ** 2))

        table.add_row("RMS Level", f"{orig_rms:.4f}", f"{enh_rms:.4f}")
        table.add_row("Duration", f"{duration:.1f}s", f"{duration:.1f}s")
        table.add_row("Timbre", "Original", self.model_type.title())

        console.print(table)


def main():
    """Main function."""
    console.print(Panel.fit(
        "[bold cyan]DDSP-Style Audio Enhancer[/bold cyan]\n"
        "Timbre Transfer & Enhancement\n"
        "[dim]Simplified implementation without ML dependencies[/dim]",
        title="🎵 Welcome"
    ))

    # === CONFIGURATION - CHANGE THESE VALUES ===
    INPUT_FILE = "/Users/nhanvu/Documents/ AI project/karaoke_auto_pipeline/output/instrumentals/eng_piano.wav"
    OUTPUT_FILE = "output/enhanced_piano_violin.wav"
    ENHANCEMENT_LEVEL = 0.7  # 0.0 = original, 1.0 = full timbre transfer
    MODEL_TYPE = "violin"  # Options: violin, flute, cello, piano
    # ==========================================

    # Check if input file exists
    if not Path(INPUT_FILE).exists():
        console.print(f"[red]Error: Input file not found: {INPUT_FILE}[/red]")
        console.print("[yellow]Please check the file path and try again.[/yellow]")
        return

    try:
        # Create enhancer
        enhancer = SimpleDDSPEnhancer()

        # Step 1: Load model
        console.print("\n[bold yellow]Step 1:[/bold yellow] Loading Timbre Profile")
        enhancer.load_model(MODEL_TYPE)

        # Step 2: Configure
        console.print("[bold yellow]Step 2:[/bold yellow] Configuring Settings")
        enhancer.configure(enhancement_level=ENHANCEMENT_LEVEL)

        # Step 3: Process
        console.print("[bold yellow]Step 3:[/bold yellow] Processing Audio")
        enhancer.process(INPUT_FILE, OUTPUT_FILE)

        console.print("\n[bold green]🎉 Success! Your audio has been enhanced with "
                      f"{MODEL_TYPE} timbre characteristics.[/bold green]")

        # Provide tips
        console.print("\n[bold cyan]💡 Tips:[/bold cyan]")
        console.print("• Try different enhancement levels (0.0 to 1.0)")
        console.print("• Experiment with different target instruments")
        console.print("• For best results, use clean instrumental recordings")

    except KeyboardInterrupt:
        console.print("\n[yellow]Process cancelled by user[/yellow]")
    except FileNotFoundError as e:
        console.print(f"\n[red]File not found: {e}[/red]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        console.print("[dim]Run with --debug flag for detailed error info[/dim]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()