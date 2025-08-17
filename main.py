#!/usr/bin/env python3
"""
High-Quality Audio Source Separation Pipeline
============================================
Ensemble approach using multiple models for optimal separation quality.
Target: SNR > 0.20
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime

import torch
import torchaudio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our ensemble separator
from modules.ensemble_separator import EnsembleSourceSeparator
from modules.audio_loader import AudioLoader

# Configure console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


class HighQualityAudioPipeline:
    """High-quality audio source separation pipeline focused on SNR > 0.20."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the high-quality separation pipeline."""
        self.config = config or self._default_config()
        self.console = console
        self._initialize_components()

    def _default_config(self) -> Dict[str, Any]:
        """Return configuration optimized for quality."""
        return {
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "sample_rate": 44100,
            "output_dir": Path("output"),
            "temp_dir": Path("temp"),

            # Quality-focused settings
            "quality_mode": "highest",
            "target_snr": 0.20,
            "timeout": 1800,  # 30 minutes for quality processing

            # Ensemble configuration
            "ensemble_models": [
                "htdemucs_6s",  # Primary high-quality model
                "htdemucs_ft",  # Fine-tuned backup
                "mdx_extra_q",  # MDX high quality (if available)
            ],
            "blend_method": "weighted_average",
            "use_frequency_weighting": True,

            # Advanced options
            "parallel_processing": True,
            "save_individual_results": False,  # Only save final ensemble result
            "detailed_metrics": True
        }

    def _initialize_components(self):
        """Initialize pipeline components."""
        self.console.print(Panel.fit(
            "[bold cyan]High-Quality Ensemble Source Separation[/bold cyan]\n"
            f"Target SNR: ≥ {self.config['target_snr']}\n"
            f"Quality Mode: {self.config['quality_mode']}\n"
            f"Device: {self.config['device']}",
            title="🎵 Quality-Focused Pipeline",
            border_style="cyan"
        ))

        # Create directories
        for dir_path in [self.config["output_dir"], self.config["temp_dir"]]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.audio_loader = AudioLoader(self.config)

        try:
            self.separator = EnsembleSourceSeparator(self.config)
            model_info = self.separator.get_model_info()

            self.console.print(f"[green]✓ Loaded {len(model_info['loaded_models'])} models:[/green]")
            for model in model_info['loaded_models']:
                weight = model_info['model_weights'].get(model, 1.0)
                self.console.print(f"  • {model} (weight: {weight})")

        except Exception as e:
            self.console.print(f"[red]Failed to initialize separator: {e}[/red]")
            raise

    def separate_audio(self, input_path: Path, output_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform high-quality source separation.

        Args:
            input_path: Path to input audio file
            output_name: Optional custom output name

        Returns:
            Processing results with quality metrics
        """
        start_time = time.time()

        if output_name is None:
            output_name = input_path.stem

        results = {
            "input": str(input_path),
            "output_name": output_name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "target_snr": self.config["target_snr"],
                "quality_mode": self.config["quality_mode"],
                "ensemble_models": self.config["ensemble_models"]
            }
        }

        try:
            # Load audio
            self._print_stage("Loading Audio", "📂")
            audio_data = self.audio_loader.load(input_path)

            self.console.print(f"[yellow]Duration: {audio_data['duration']:.1f}s, "
                               f"Channels: {audio_data['channels']}, "
                               f"Sample Rate: {audio_data['sample_rate']}Hz[/yellow]")

            # Perform ensemble separation with progress tracking
            self._print_stage("Ensemble Source Separation", "🎛️")

            with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=self.console
            ) as progress:

                task = progress.add_task("Processing...", total=100)

                def progress_callback(percent: float, message: str):
                    progress.update(task, completed=percent * 100, description=message)

                separated = self.separator.separate(
                    audio_data["waveform"],
                    audio_data["sample_rate"],
                    progress_callback=progress_callback
                )

            # Save separated stems
            self._print_stage("Saving Results", "💾")
            stem_paths = self._save_stems(separated, audio_data, output_name)

            # Compile results
            results.update({
                "stems": stem_paths,
                "metrics": separated["metrics"],
                "ensemble_info": separated["ensemble_info"],
                "processing_time": separated["processing_time"],
                "total_time": time.time() - start_time
            })

            # Save processing results
            self._save_processing_results(results)

            # Display quality assessment
            self._display_quality_results(results)

            return results

        except Exception as e:
            logger.error(f"Separation failed: {str(e)}")
            results["error"] = str(e)
            raise

    def _print_stage(self, stage: str, emoji: str):
        """Print a processing stage header."""
        self.console.print(f"\n{emoji} [bold cyan]{stage}[/bold cyan]")

    def _save_stems(self, separated: Dict, audio_data: Dict, output_name: str) -> Dict[str, str]:
        """Save separated stems to disk."""
        output_dir = self.config["output_dir"] / "separated" / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        stem_paths = {}

        for stem_name, stem_audio in separated["stems"].items():
            stem_path = output_dir / f"{stem_name}.wav"

            # Ensure proper format for saving
            if stem_audio.dim() == 1:
                stem_audio = stem_audio.unsqueeze(0)

            torchaudio.save(
                stem_path,
                stem_audio,
                audio_data["sample_rate"],
                bits_per_sample=24  # High quality output
            )

            stem_paths[stem_name] = str(stem_path)

            # Calculate and display stem info
            duration = stem_audio.shape[-1] / audio_data["sample_rate"]
            size_mb = stem_path.stat().st_size / (1024 * 1024)

            self.console.print(f"  ✓ {stem_name}: {duration:.1f}s, {size_mb:.1f}MB")

        return stem_paths

    def _save_processing_results(self, results: Dict):
        """Save detailed processing results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.config["output_dir"] / f"separation_results_{timestamp}.json"

        # Prepare results for JSON serialization
        json_results = json.loads(json.dumps(results, default=str))

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        self.console.print(f"[dim]Results saved to {results_file}[/dim]")

    def _display_quality_results(self, results: Dict):
        """Display comprehensive quality assessment."""
        metrics = results["metrics"]

        # Main quality table
        table = Table(title="Separation Quality Assessment", show_header=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="yellow", width=15)
        table.add_column("Status", style="white", width=15)

        # SNR assessment
        snr = metrics.get("reconstruction_snr", 0)
        snr_status = "✓ Excellent" if snr >= 0.30 else "✓ Good" if snr >= 0.20 else "⚠ Fair" if snr >= 0.10 else "✗ Poor"
        snr_color = "green" if snr >= 0.20 else "yellow" if snr >= 0.10 else "red"

        table.add_row("SNR", f"{snr:.3f} dB", f"[{snr_color}]{snr_status}[/{snr_color}]")
        table.add_row("Quality Grade", metrics.get("quality_grade", "Unknown"),
                      "✓ Target Met" if metrics.get("target_achieved", False) else "✗ Target Missed")
        table.add_row("Stems Generated", str(metrics.get("num_stems", 0)), "")
        table.add_row("Processing Time", f"{results['processing_time']:.1f}s", "")

        self.console.print("\n")
        self.console.print(table)

        # Stem energy distribution
        stem_table = Table(title="Stem Energy Distribution", show_header=True)
        stem_table.add_column("Stem", style="cyan")
        stem_table.add_column("Energy Ratio", style="yellow")
        stem_table.add_column("Quality", style="green")

        for stem_name in results["stems"].keys():
            energy_ratio = metrics.get(f"{stem_name}_energy_ratio", 0)
            energy_percent = energy_ratio * 100

            # Simple quality assessment based on energy
            if energy_percent > 15:
                quality = "High"
            elif energy_percent > 5:
                quality = "Medium"
            else:
                quality = "Low"

            stem_table.add_row(stem_name.capitalize(), f"{energy_percent:.1f}%", quality)

        self.console.print(stem_table)

        # Ensemble information
        ensemble_info = results.get("ensemble_info", {})
        if ensemble_info:
            self.console.print(f"\n[bold]Ensemble Details:[/bold]")
            self.console.print(f"Models Used: {', '.join(ensemble_info.get('models_used', []))}")
            self.console.print(f"Blend Method: {ensemble_info.get('blend_method', 'Unknown')}")
            self.console.print(f"Quality Mode: {ensemble_info.get('quality_mode', 'Unknown')}")


def validate_input_file(file_path: Path) -> bool:
    """Validate input audio file."""
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        return False

    # Check file extension
    valid_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    if file_path.suffix.lower() not in valid_extensions:
        console.print(f"[red]Error: Unsupported file format: {file_path.suffix}[/red]")
        console.print(f"Supported formats: {', '.join(valid_extensions)}")
        return False

    # Check file size (warn if very large)
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > 500:  # 500MB warning
        console.print(f"[yellow]Warning: Large file ({size_mb:.1f}MB). Processing may take a long time.[/yellow]")

    return True


def check_system_requirements():
    """Check system requirements for high-quality processing."""
    console.print(Panel.fit(
        "[bold cyan]System Requirements Check[/bold cyan]",
        title="🔧 System Check"
    ))

    # Check PyTorch
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[green]✓ PyTorch available[/green] (Device: {device})")
    except ImportError:
        console.print("[red]✗ PyTorch not available[/red]")
        return False

    # Check Demucs
    try:
        from demucs import pretrained
        console.print("[green]✓ Demucs available[/green]")
    except ImportError:
        console.print("[red]✗ Demucs not available[/red]")
        console.print("Install with: pip install demucs")
        return False

    # Check memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        if memory_gb >= 16:
            console.print(f"[green]✓ Memory: {memory_gb:.1f}GB[/green]")
        else:
            console.print(f"[yellow]⚠ Memory: {memory_gb:.1f}GB (16GB+ recommended)[/yellow]")
    except ImportError:
        console.print("[dim]Memory check skipped (psutil not installed)[/dim]")

    return True


def main():
    """Main entry point for high-quality audio separation."""
    parser = argparse.ArgumentParser(
        description="High-Quality Audio Source Separation (Target SNR > 0.20)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py song.wav                    # Separate with default settings
  python main.py song.mp3 --output my_song  # Custom output name
  python main.py song.wav --models htdemucs_6s htdemucs_ft  # Specific models
  python main.py --check                    # Check system requirements
        """
    )

    parser.add_argument("input", nargs="?", type=Path, help="Input audio file")
    parser.add_argument("--output", "-o", help="Output name (default: input filename)")
    parser.add_argument("--models", nargs="+",
                        choices=["htdemucs_6s", "htdemucs_ft", "htdemucs", "mdx_extra_q", "mdx23c_musdb18"],
                        help="Specific models to use in ensemble")
    parser.add_argument("--target-snr", type=float, default=0.20,
                        help="Target SNR threshold (default: 0.20)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Processing timeout in seconds (default: 1800)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto", help="Processing device")
    parser.add_argument("--check", action="store_true",
                        help="Check system requirements and exit")
    parser.add_argument("--quality", choices=["highest", "high", "balanced"],
                        default="highest", help="Quality mode")

    args = parser.parse_args()

    # System requirements check
    if args.check:
        success = check_system_requirements()
        sys.exit(0 if success else 1)

    if not args.input:
        parser.print_help()
        return

    # Validate input
    if not validate_input_file(args.input):
        sys.exit(1)

    # Check system requirements
    if not check_system_requirements():
        console.print("[red]System requirements not met![/red]")
        sys.exit(1)

    # Print banner
    console.print(Panel.fit(
        f"[bold magenta]High-Quality Audio Source Separation[/bold magenta]\n"
        f"📂 Input: {args.input}\n"
        f"🎯 Target SNR: ≥ {args.target_snr}\n"
        f"⚙️ Quality: {args.quality}\n"
        f"🔧 Device: {args.device}",
        title="🎵 Ensemble Separation",
        border_style="magenta"
    ))

    # Configure pipeline
    config = {
        "device": args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"),
        "output_dir": Path("output"),
        "temp_dir": Path("temp"),
        "quality_mode": args.quality,
        "target_snr": args.target_snr,
        "timeout": args.timeout,
        "detailed_metrics": True
    }

    if args.models:
        config["ensemble_models"] = args.models

    # Run pipeline
    try:
        pipeline = HighQualityAudioPipeline(config)
        results = pipeline.separate_audio(args.input, args.output)

        # Final summary
        snr = results["metrics"]["reconstruction_snr"]
        target_met = snr >= args.target_snr

        console.print(f"\n[bold green]✨ Separation Complete![/bold green]")
        console.print(f"SNR: {snr:.3f} dB {'✓' if target_met else '✗'}")
        console.print(f"Total Time: {results['total_time']:.1f}s")
        console.print(f"Output Directory: output/separated/{args.output or args.input.stem}/")

        if target_met:
            console.print("[bold green]🎉 Target SNR achieved![/bold green]")
        else:
            console.print(f"[yellow]⚠ Target SNR ({args.target_snr}) not reached[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()