#!/usr/bin/env python3
"""
High-Quality Audio Source Separation Pipeline
============================================
Production-ready ensemble approach for optimal separation quality.
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

# Import our modules
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


class AudioSeparationPipeline:
    """Production audio source separation pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the separation pipeline."""
        # Merge provided config with defaults
        self.config = self._default_config()
        if config:
            self.config.update(config)
        self.console = console
        self._initialize_components()

    def _default_config(self) -> Dict[str, Any]:
        """Return optimized configuration."""
        device = "auto"
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        return {
            "device": device,
            "sample_rate": 44100,
            "output_dir": Path("output"),
            "temp_dir": Path("temp"),

            # Quality settings
            "quality_mode": "highest",
            "target_snr": 0.20,
            "timeout": 1800,

            # Ensemble configuration
            "ensemble_models": None,  # Will use defaults based on quality_mode
            "blend_method": "weighted_average",
            "use_frequency_weighting": True,

            # Performance
            "max_parallel_models": 2,
            "enable_cache": True,

            # Output
            "output_format": "wav",
            "output_bitdepth": 24,
            "save_metrics": True
        }

    def _initialize_components(self):
        """Initialize pipeline components."""
        # Create directories
        self.config["output_dir"].mkdir(parents=True, exist_ok=True)
        self.config["temp_dir"].mkdir(parents=True, exist_ok=True)

        # Clean temp directory
        self._clean_temp_dir()

        # Initialize components
        self.audio_loader = AudioLoader(self.config)

        try:
            self.separator = EnsembleSourceSeparator(self.config)
            self._print_initialization_info()
        except Exception as e:
            self.console.print(f"[red]Failed to initialize separator: {e}[/red]")
            raise

    def _clean_temp_dir(self):
        """Clean temporary directory."""
        temp_dir = self.config["temp_dir"]
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                try:
                    if file.is_file():
                        file.unlink()
                except Exception:
                    pass

    def _print_initialization_info(self):
        """Print initialization information."""
        model_info = self.separator.get_model_info()

        self.console.print(Panel.fit(
            f"[bold cyan]Audio Source Separation Pipeline[/bold cyan]\n"
            f"Device: {model_info['device']}\n"
            f"Quality Mode: {model_info['quality_mode']}\n"
            f"Target SNR: ≥ {model_info['target_snr']:.2f} dB\n"
            f"Models: {len(model_info['loaded_models'])}",
            title="🎵 Pipeline Ready",
            border_style="cyan"
        ))

        # Print loaded models
        self.console.print("\n[bold]Loaded Models:[/bold]")
        for model_name in model_info['loaded_models']:
            config = model_info['model_configs'][model_name]
            self.console.print(
                f"  • {model_name}: "
                f"weight={config['weight']:.2f}, "
                f"stems={config['stems']}, "
                f"quality={config['quality_score']:.2f}"
            )

    def separate_audio(self,
                       input_path: Path,
                       output_name: Optional[str] = None,
                       save_individual_stems: bool = True) -> Dict[str, Any]:
        """
        Perform source separation on audio file.

        Args:
            input_path: Path to input audio file
            output_name: Optional output name (defaults to input filename)
            save_individual_stems: Whether to save individual stem files

        Returns:
            Processing results dictionary
        """
        start_time = time.time()

        if output_name is None:
            output_name = input_path.stem

        results = {
            "input": str(input_path),
            "output_name": output_name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "quality_mode": self.config["quality_mode"],
                "target_snr": self.config["target_snr"],
                "device": str(self.config["device"])
            }
        }

        try:
            # Load audio
            self._print_stage("Loading Audio", "📂")
            audio_data = self.audio_loader.load(input_path)

            self.console.print(
                f"Duration: {audio_data['duration']:.1f}s | "
                f"Channels: {audio_data['channels']} | "
                f"Sample Rate: {audio_data['sample_rate']}Hz | "
                f"Format: {audio_data['format'].upper()}"
            )

            # Check duration
            if audio_data['duration'] > 600:  # 10 minutes
                self.console.print("[yellow]⚠ Long audio file. Processing may take several minutes.[/yellow]")

            # Perform separation
            self._print_stage("Running Ensemble Separation", "🎛️")

            with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=self.console,
                    transient=False
            ) as progress:

                task = progress.add_task("Initializing...", total=100)

                def progress_callback(percent: float, message: str):
                    progress.update(task, completed=percent * 100, description=message)

                separated = self.separator.separate(
                    audio_data["waveform"],
                    audio_data["sample_rate"],
                    progress_callback=progress_callback
                )

            # Save stems
            if save_individual_stems:
                self._print_stage("Saving Separated Stems", "💾")
                stem_paths = self._save_stems(separated, audio_data, output_name)
                results["stems"] = stem_paths

            # Update results
            results.update({
                "metrics": separated["metrics"],
                "ensemble_info": separated["ensemble_info"],
                "processing_time": separated["processing_time"],
                "total_time": time.time() - start_time
            })

            # Save metrics if enabled
            if self.config["save_metrics"]:
                self._save_metrics(results, output_name)

            # Display results
            self._display_results(results)

            return results

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Processing interrupted by user[/yellow]")
            raise
        except Exception as e:
            logger.error(f"Separation failed: {str(e)}")
            results["error"] = str(e)
            results["total_time"] = time.time() - start_time
            raise

    def _print_stage(self, stage: str, emoji: str):
        """Print processing stage."""
        self.console.print(f"\n{emoji} [bold cyan]{stage}[/bold cyan]")

    def _save_stems(self, separated: Dict, audio_data: Dict, output_name: str) -> Dict[str, str]:
        """Save separated stems to disk."""
        output_dir = self.config["output_dir"] / "separated" / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        stem_paths = {}

        # Save each stem
        for stem_name, stem_audio in separated["stems"].items():
            stem_path = output_dir / f"{stem_name}.{self.config['output_format']}"

            # Ensure proper shape
            if stem_audio.dim() == 1:
                stem_audio = stem_audio.unsqueeze(0)

            # Save with high quality
            if self.config["output_format"] == "wav":
                torchaudio.save(
                    stem_path,
                    stem_audio,
                    audio_data["sample_rate"],
                    bits_per_sample=self.config["output_bitdepth"]
                )
            else:
                torchaudio.save(stem_path, stem_audio, audio_data["sample_rate"])

            stem_paths[stem_name] = str(stem_path)

            # Display info
            duration = stem_audio.shape[-1] / audio_data["sample_rate"]
            size_mb = stem_path.stat().st_size / (1024 * 1024)
            self.console.print(f"  ✓ {stem_name}: {duration:.1f}s, {size_mb:.1f}MB")

        return stem_paths

    def _save_metrics(self, results: Dict, output_name: str):
        """Save processing metrics."""
        metrics_dir = self.config["output_dir"] / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = metrics_dir / f"{output_name}_{timestamp}.json"

        # Prepare for JSON serialization
        json_results = json.loads(json.dumps(results, default=str))

        with open(metrics_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        self.console.print(f"[dim]Metrics saved to {metrics_file}[/dim]")

    def _display_results(self, results: Dict):
        """Display separation results."""
        metrics = results["metrics"]

        # Quality metrics table
        table = Table(title="Separation Quality Metrics", show_header=True)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="yellow", width=20)
        table.add_column("Status", style="white", width=20)

        # SNR
        snr = metrics.get("reconstruction_snr", 0)
        snr_color = "green" if snr >= 0.20 else "yellow" if snr >= 0.10 else "red"
        snr_status = "✓ Target Met" if snr >= self.config["target_snr"] else "✗ Below Target"
        table.add_row("SNR", f"{snr:.3f} dB", f"[{snr_color}]{snr_status}[/{snr_color}]")

        # SDR if available
        if "sdr" in metrics:
            sdr = metrics["sdr"]
            table.add_row("SDR", f"{sdr:.3f} dB", "")

        # Quality grade
        table.add_row("Quality Grade", metrics.get("quality_grade", "Unknown"), "")

        # Processing info
        table.add_row("Stems Generated", str(metrics.get("num_stems", 0)), "")
        table.add_row("Processing Time", f"{results['processing_time']:.1f}s", "")
        table.add_row("Total Time", f"{results['total_time']:.1f}s", "")

        self.console.print("\n")
        self.console.print(table)

        # Stem energy distribution
        if any(k.endswith("_energy_ratio") for k in metrics.keys()):
            stem_table = Table(title="Stem Energy Distribution", show_header=True)
            stem_table.add_column("Stem", style="cyan")
            stem_table.add_column("Energy Ratio", style="yellow")
            stem_table.add_column("Level", style="green")

            for key in sorted(metrics.keys()):
                if key.endswith("_energy_ratio"):
                    stem_name = key.replace("_energy_ratio", "")
                    ratio = metrics[key]
                    percent = ratio * 100

                    if percent > 20:
                        level = "High"
                    elif percent > 5:
                        level = "Medium"
                    else:
                        level = "Low"

                    stem_table.add_row(stem_name.capitalize(), f"{percent:.1f}%", level)

            self.console.print(stem_table)

    def batch_process(self, input_files: list[Path], output_dir: Optional[Path] = None) -> list[Dict]:
        """
        Process multiple audio files.

        Args:
            input_files: List of input file paths
            output_dir: Optional output directory

        Returns:
            List of results for each file
        """
        if output_dir:
            self.config["output_dir"] = output_dir

        results = []
        total_files = len(input_files)

        self.console.print(f"\n[bold]Batch Processing {total_files} files[/bold]\n")

        for i, input_file in enumerate(input_files, 1):
            self.console.print(f"\n[bold magenta]File {i}/{total_files}: {input_file.name}[/bold magenta]")
            self.console.print("=" * 60)

            try:
                result = self.separate_audio(input_file)
                results.append(result)
                self.console.print(f"[green]✓ Completed {input_file.name}[/green]")
            except Exception as e:
                self.console.print(f"[red]✗ Failed {input_file.name}: {e}[/red]")
                results.append({"input": str(input_file), "error": str(e)})

        # Summary
        successful = sum(1 for r in results if "error" not in r)
        self.console.print(f"\n[bold]Batch Complete: {successful}/{total_files} successful[/bold]")

        return results


def validate_input(file_path: Path) -> bool:
    """Validate input file."""
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        return False

    valid_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wma'}
    if file_path.suffix.lower() not in valid_extensions:
        console.print(f"[red]Error: Unsupported format: {file_path.suffix}[/red]")
        console.print(f"Supported: {', '.join(sorted(valid_extensions))}")
        return False

    # Check file size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > 1000:  # 1GB warning
        response = console.input(f"[yellow]Large file ({size_mb:.1f}MB). Continue? (y/n): [/yellow]")
        if response.lower() != 'y':
            return False

    return True


def check_requirements():
    """Check system requirements."""
    issues = []

    # Check PyTorch
    try:
        import torch
        device_info = "CPU"
        if torch.backends.mps.is_available():
            device_info = "Apple Silicon (MPS)"
        elif torch.cuda.is_available():
            device_info = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
        console.print(f"[green]✓ PyTorch available[/green] - Device: {device_info}")
    except ImportError:
        issues.append("PyTorch not installed (pip install torch)")

    # Check Demucs
    try:
        import demucs
        console.print("[green]✓ Demucs available[/green]")
    except ImportError:
        issues.append("Demucs not installed (pip install demucs)")

    # Check memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        status = "green" if memory_gb >= 16 else "yellow"
        console.print(f"[{status}]✓ Memory: {memory_gb:.1f}GB total, {available_gb:.1f}GB available[/{status}]")
        if memory_gb < 8:
            issues.append("Low memory (8GB minimum, 16GB recommended)")
    except ImportError:
        console.print("[dim]Memory check skipped (psutil not installed)[/dim]")

    return len(issues) == 0, issues


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="High-Quality Audio Source Separation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s song.mp3                     # Separate with default settings
  %(prog)s song.wav -o my_song          # Custom output name
  %(prog)s song.mp3 -q high             # High quality mode
  %(prog)s *.mp3 --batch                # Process multiple files
  %(prog)s --check                      # Check system requirements
        """
    )

    parser.add_argument("input", nargs="*", type=Path, help="Input audio file(s)")
    parser.add_argument("-o", "--output", help="Output name")
    parser.add_argument("-q", "--quality",
                        choices=["highest", "high", "balanced", "fast"],
                        default="highest", help="Quality mode (default: highest)")
    parser.add_argument("--snr-target", type=float, default=0.20,
                        help="Target SNR threshold (default: 0.20)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto", help="Processing device")
    parser.add_argument("--models", nargs="+",
                        help="Specific models to use")
    parser.add_argument("--blend", choices=["weighted_average", "median", "frequency_weighted"],
                        default="weighted_average", help="Blending method")
    parser.add_argument("--batch", action="store_true",
                        help="Process multiple files")
    parser.add_argument("--no-stems", action="store_true",
                        help="Don't save individual stem files")
    parser.add_argument("--check", action="store_true",
                        help="Check system requirements")

    args = parser.parse_args()

    # Check requirements
    if args.check:
        console.print(Panel.fit("[bold]System Requirements Check[/bold]", border_style="cyan"))
        success, issues = check_requirements()
        if not success:
            console.print("\n[red]Issues found:[/red]")
            for issue in issues:
                console.print(f"  • {issue}")
        sys.exit(0 if success else 1)

    # Validate input
    if not args.input:
        parser.print_help()
        sys.exit(1)

    # Batch processing
    if args.batch or len(args.input) > 1:
        valid_files = [f for f in args.input if validate_input(f)]
        if not valid_files:
            console.print("[red]No valid input files[/red]")
            sys.exit(1)
    else:
        if not validate_input(args.input[0]):
            sys.exit(1)
        valid_files = [args.input[0]]

    # Check requirements
    success, issues = check_requirements()
    if not success:
        console.print("[red]System requirements not met:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")
        response = console.input("[yellow]Continue anyway? (y/n): [/yellow]")
        if response.lower() != 'y':
            sys.exit(1)

    # Configure pipeline
    config = {
        "quality_mode": args.quality,
        "target_snr": args.snr_target,
        "device": args.device,
        "blend_method": args.blend,
        "output_dir": Path("output"),  # Add default output directory
        "temp_dir": Path("temp")  # Add default temp directory
    }

    if args.models:
        config["ensemble_models"] = args.models

    # Print header
    console.print(Panel.fit(
        f"[bold magenta]Audio Source Separation[/bold magenta]\n"
        f"Files: {len(valid_files)}\n"
        f"Quality: {args.quality}\n"
        f"Target SNR: ≥ {args.snr_target:.2f} dB",
        title="🎵 Processing Started",
        border_style="magenta"
    ))

    # Run pipeline
    try:
        pipeline = AudioSeparationPipeline(config)

        if len(valid_files) > 1:
            results = pipeline.batch_process(valid_files)
        else:
            results = [pipeline.separate_audio(
                valid_files[0],
                args.output,
                save_individual_stems=not args.no_stems
            )]

        # Final summary
        successful = sum(1 for r in results if "error" not in r)
        console.print(f"\n[bold green]✨ Processing Complete![/bold green]")
        console.print(f"Successful: {successful}/{len(results)}")

        if successful > 0:
            avg_snr = sum(r["metrics"]["reconstruction_snr"] for r in results if "error" not in r) / successful
            console.print(f"Average SNR: {avg_snr:.3f} dB")
            # Use pipeline's config for output_dir
            console.print(f"Output: {pipeline.config['output_dir']}/separated/")

    except KeyboardInterrupt:
        console.print("\n[yellow]Processing cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()