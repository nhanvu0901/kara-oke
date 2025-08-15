import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
import logging
import torchaudio
logger = logging.getLogger(__name__)


class AudioVisualizer:
    """Create educational visualizations of audio data."""

    def __init__(self, config: Dict):
        self.config = config
        plt.style.use('seaborn-v0_8-darkgrid')

    def create_spectrogram(self,
                           waveform: torch.Tensor,
                           sample_rate: int,
                           output_path: Path,
                           title: str = "Spectrogram") -> None:
        """Create and save a spectrogram visualization."""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Waveform
            if waveform.dim() > 1:
                wave_data = waveform.mean(dim=0)  # Convert to mono for visualization
            else:
                wave_data = waveform

            time_axis = np.linspace(0, len(wave_data) / sample_rate, len(wave_data))
            axes[0].plot(time_axis, wave_data.numpy(), linewidth=0.5)
            axes[0].set_xlabel("Time (s)")
            axes[0].set_ylabel("Amplitude")
            axes[0].set_title(f"{title} - Waveform")
            axes[0].grid(True, alpha=0.3)

            # Spectrogram
            spectrogram = torchaudio.transforms.Spectrogram()(wave_data)
            log_spec = torch.log10(spectrogram + 1e-10)

            img = axes[1].imshow(
                log_spec.numpy(),
                aspect='auto',
                origin='lower',
                cmap='viridis',
                extent=[0, len(wave_data) / sample_rate, 0, sample_rate / 2]
            )
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Frequency (Hz)")
            axes[1].set_title(f"{title} - Spectrogram")

            plt.colorbar(img, ax=axes[1], label="Log Magnitude")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Visualization saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")

    def create_combined_plot(self,
                             waveform: torch.Tensor,
                             sample_rate: int,
                             output_path: Path) -> None:
        """Create a comprehensive audio analysis plot."""
        try:
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))

            if waveform.dim() > 1:
                wave_data = waveform.mean(dim=0)
            else:
                wave_data = waveform

            # 1. Waveform
            time_axis = np.linspace(0, len(wave_data) / sample_rate, len(wave_data))
            axes[0, 0].plot(time_axis, wave_data.numpy(), linewidth=0.5, color='blue')
            axes[0, 0].set_title("Waveform")
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].set_ylabel("Amplitude")
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Spectrogram
            spec_transform = torchaudio.transforms.Spectrogram(n_fft=2048)
            spectrogram = spec_transform(wave_data)
            log_spec = torch.log10(spectrogram + 1e-10)

            axes[0, 1].imshow(
                log_spec.numpy(),
                aspect='auto',
                origin='lower',
                cmap='magma'
            )
            axes[0, 1].set_title("Spectrogram")
            axes[0, 1].set_xlabel("Time (frames)")
            axes[0, 1].set_ylabel("Frequency (bins)")

            # 3. Mel Spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=128
            )
            mel_spec = mel_transform(wave_data)
            log_mel = torch.log10(mel_spec + 1e-10)

            axes[1, 0].imshow(
                log_mel.numpy(),
                aspect='auto',
                origin='lower',
                cmap='coolwarm'
            )
            axes[1, 0].set_title("Mel Spectrogram")
            axes[1, 0].set_xlabel("Time (frames)")
            axes[1, 0].set_ylabel("Mel Bands")

            # 4. Energy over time
            window_size = 2048
            hop_length = 512
            energy = []
            for i in range(0, len(wave_data) - window_size, hop_length):
                window = wave_data[i:i + window_size]
                energy.append(torch.sum(window ** 2).item())

            energy_time = np.linspace(0, len(wave_data) / sample_rate, len(energy))
            axes[1, 1].plot(energy_time, energy, color='green')
            axes[1, 1].set_title("Energy Over Time")
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Energy")
            axes[1, 1].grid(True, alpha=0.3)

            # 5. Frequency spectrum (FFT)
            fft = torch.fft.rfft(wave_data)
            magnitude = torch.abs(fft)
            freqs = torch.fft.rfftfreq(len(wave_data), 1 / sample_rate)

            axes[2, 0].plot(freqs[:len(freqs) // 2].numpy(),
                            magnitude[:len(magnitude) // 2].numpy(),
                            color='purple')
            axes[2, 0].set_title("Frequency Spectrum")
            axes[2, 0].set_xlabel("Frequency (Hz)")
            axes[2, 0].set_ylabel("Magnitude")
            axes[2, 0].set_xlim([0, sample_rate / 4])
            axes[2, 0].grid(True, alpha=0.3)

            # 6. Phase spectrum
            phase = torch.angle(fft)
            axes[2, 1].plot(freqs[:1000].numpy(), phase[:1000].numpy(),
                            color='orange', linewidth=0.5)
            axes[2, 1].set_title("Phase Spectrum (Low Freq)")
            axes[2, 1].set_xlabel("Frequency (Hz)")
            axes[2, 1].set_ylabel("Phase (radians)")
            axes[2, 1].grid(True, alpha=0.3)

            plt.suptitle("Comprehensive Audio Analysis", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Combined visualization saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to create combined plot: {e}")
