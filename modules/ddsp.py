#!/usr/bin/env python3
"""
DDSP Audio Enhancement - Simplified Version
Enhance audio using DDSP for timbre transfer and quality improvement.
"""

import time
import torch
import torchaudio
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# DDSP imports
try:
    from ddsp_pytorch.ddsp import mlp, gru, extract_loudness, extract_pitch
    DDSP_AVAILABLE = True
except ImportError:
    DDSP_AVAILABLE = False
    print("Please install ddsp_pytorch dependencies: pip install -r ../ddsp_pytorch/requirements.txt")

console = Console()


class DDSPEnhancer:
    """Simple audio enhancer using DDSP for timbre transfer."""

    def __init__(self):
        self.model = None
        self.sample_rate = 16000  # DDSP standard rate
        self.device = "cpu"  # Use CPU for M4 Mac compatibility

    def load_model(self, model_type="violin"):
        """Load pre-trained DDSP model."""
        console.print(Panel.fit(
            "[bold cyan]Loading DDSP Model[/bold cyan]\n"
            f"Model: [yellow]{model_type}[/yellow]\n"
            "[dim]This may take a moment on first run[/dim]",
            title="🎵 Model Loading"
        ))

        if not DDSP_AVAILABLE:
            raise RuntimeError("DDSP not installed. Run: pip install ddsp_pytorch")

        # Load pre-trained model
        console.print(f"Loading DDSP {model_type} model...")

        # Assuming user has downloaded the pretrained .ts file from:
        # Violin: https://nubo.ircam.fr/index.php/s/f6XB4Kp9onxiNwZ/download
        # Rename it to ddsp_pretrained_violin.ts or adjust the path below
        model_path = f"ddsp_pretrained_{model_type}.ts"
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Pretrained model file not found: {model_path}. Download from IRCAM repository.")

        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.model.to(self.device)

        console.print(f"[green]✓[/green] Model ready! Sample rate: {self.sample_rate} Hz\n")

    def configure(self, enhancement_level=0.5):
        """Configure enhancement settings."""
        console.print("[bold]Configuring enhancement settings...[/bold]")

        self.enhancement_level = enhancement_level

        # Display settings
        table = Table(title="DDSP Settings")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_row("Sample Rate", f"{self.sample_rate} Hz")
        table.add_row("Enhancement Level", f"{enhancement_level:.1%}")
        table.add_row("Processing Mode", "Timbre Transfer")
        table.add_row("Device", self.device.upper())
        console.print(table)
        console.print("[green]✓[/green] Settings applied\n")

    def extract_features(self, audio):
        """Extract F0 (pitch) and loudness from audio using DDSP tools."""
        console.print("Extracting audio features...")

        # Convert to numpy for processing
        if isinstance(audio, torch.Tensor):
            audio_np = audio.numpy().squeeze()
        else:
            audio_np = audio

        # Use DDSP's extract_pitch and extract_loudness
        # Assuming these functions take numpy array and sample_rate, return arrays of shape (frames,)
        f0 = extract_pitch(audio_np, self.sample_rate)
        loudness = extract_loudness(audio_np, self.sample_rate)

        console.print(f"[green]✓[/green] Features extracted: {len(f0)} frames\n")

        return {
            'f0': f0,
            'loudness': loudness,
            'audio': audio_np
        }

    def enhance_audio(self, features):
        """Apply DDSP timbre transfer to audio features."""
        console.print("Applying DDSP enhancement...")

        f0 = features['f0']
        loudness = features['loudness']
        original_audio = features['audio']

        # Convert to tensors: (1, frames, 1)
        f0_tensor = torch.from_numpy(f0).float().to(self.device).unsqueeze(0).unsqueeze(-1)
        loudness_tensor = torch.from_numpy(loudness).float().to(self.device).unsqueeze(0).unsqueeze(-1)

        # Synthesize with DDSP model
        with torch.no_grad():
            synth_audio = self.model(f0_tensor, loudness_tensor).squeeze(0).cpu().numpy()

        # Mix with original based on enhancement_level (0: original, 1: full transfer)
        enhanced_audio = (1 - self.enhancement_level) * original_audio + self.enhancement_level * synth_audio

        # Normalize to prevent clipping
        max_val = np.max(np.abs(enhanced_audio))
        if max_val > 0.95:
            enhanced_audio = enhanced_audio * (0.95 / max_val)

        console.print(f"[green]✓[/green] Enhancement applied\n")

        return enhanced_audio

    def process(self, input_file, output_file):
        """Process audio file with DDSP enhancement."""

        # Step 1: Load audio
        console.print(f"[bold]Loading audio file:[/bold] {Path(input_file).name}")
        audio, original_sr = torchaudio.load(input_file)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            console.print("Converting stereo to mono...")
            audio = audio.mean(dim=0, keepdim=True)

        # Resample to DDSP sample rate
        if original_sr != self.sample_rate:
            console.print(f"Resampling from {original_sr} Hz to {self.sample_rate} Hz...")
            resampler = torchaudio.transforms.Resample(original_sr, self.sample_rate)
            audio = resampler(audio)

        duration = audio.shape[1] / self.sample_rate
        console.print(f"[green]✓[/green] Audio ready: {duration:.1f} seconds\n")

        # Step 2: Extract features
        console.print(Panel.fit(
            "[bold magenta]DDSP Processing[/bold magenta]\n"
            f"Mode: Timbre Enhancement",
            title="🎨 Processing"
        ))

        features = self.extract_features(audio)

        # Step 3: Enhance with DDSP
        start_time = time.time()
        enhanced_audio = self.enhance_audio(features)
        elapsed = time.time() - start_time

        console.print(f"Processing completed in {elapsed:.1f} seconds")

        # Step 4: Save output
        console.print("Saving enhanced audio...")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert back to tensor for saving
        enhanced_tensor = torch.from_numpy(enhanced_audio).float().unsqueeze(0)

        # Resample back to original sample rate if needed
        if original_sr != self.sample_rate:
            console.print(f"Resampling back to {original_sr} Hz...")
            resampler = torchaudio.transforms.Resample(self.sample_rate, original_sr)
            enhanced_tensor = resampler(enhanced_tensor)

        torchaudio.save(str(output_path), enhanced_tensor, original_sr)

        # Show results
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        console.print(f"\n[bold green]✨ Enhancement Complete![/bold green]")
        console.print(f"Output: {output_path.name} ({file_size_mb:.2f} MB)")
        console.print(f"Location: {output_path.absolute()}\n")


def main():
    """Main function."""
    console.print(Panel.fit(
        "[bold cyan]DDSP Audio Enhancer[/bold cyan]\n"
        "Timbre Transfer & Enhancement",
        title="🎵 Welcome"
    ))

    # === CONFIGURATION - CHANGE THESE VALUES ===
    INPUT_FILE = "/Users/nhanvu/Documents/AI project/karaoke_auto_pipeline/output/instrumentals/eng_piano.wav"
    OUTPUT_FILE = "enhanced_piano_ddsp.wav"
    ENHANCEMENT_LEVEL = 0.8  # 0.0 = no change, 1.0 = maximum enhancement
    MODEL_TYPE = "violin"  # Options: violin, saxophone (download pretrained .ts files from IRCAM)
    # ==========================================

    try:
        # Create enhancer
        enhancer = DDSPEnhancer()

        # Step 1: Load model
        console.print("\n[bold yellow]Step 1:[/bold yellow] Loading DDSP Model")
        enhancer.load_model(MODEL_TYPE)

        # Step 2: Configure
        console.print("[bold yellow]Step 2:[/bold yellow] Configuring Settings")
        enhancer.configure(enhancement_level=ENHANCEMENT_LEVEL)

        # Step 3: Process
        console.print("[bold yellow]Step 3:[/bold yellow] Processing Audio")
        enhancer.process(INPUT_FILE, OUTPUT_FILE)

        console.print("[bold green]🎉 Done! Your enhanced audio is ready.[/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Process cancelled[/yellow]")
    except FileNotFoundError:
        console.print(f"\n[red]Error: Input file not found: '{INPUT_FILE}'[/red]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()