#!/usr/bin/env python3
"""
Audio Processing Pipeline with MVSEP Integration
================================================
Audio source separation using MVSEP API.
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
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules
from modules.audio_loader import AudioLoader
from modules.mvsep_separator import MVSEPSeparator
from modules.audio_analyzer import AudioAnalyzer
from modules.visualization import AudioVisualizer

# Configure console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


class AudioSeparationPipeline:
    """Main pipeline for audio source separation using MVSEP."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the audio separation pipeline."""
        self.config = config or self._default_config()
        self.console = console

        # Verify API key
        if not os.environ.get("MVSEP_API_KEY"):
            self.console.print("[bold red]ERROR: MVSEP_API_KEY not found in environment![/bold red]")
            self.console.print("Please set it in your .env file or export it:")
            self.console.print("export MVSEP_API_KEY='your-api-key-here'")
            sys.exit(1)

        self._initialize_components()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "sample_rate": 44100,
            "output_dir": Path("output"),
            "temp_dir": Path("temp"),
            "visualization": False,
            "mvsep": {
                "api_timeout": 600,
                "quality": "high"
            }
        }

    def _initialize_components(self):
        """Initialize pipeline components."""
        self.console.print(Panel.fit(
            "[bold cyan]Initializing Audio Separation Pipeline[/bold cyan]\n"
            "Connecting to MVSEP API...",
            title="🎵 Pipeline Initialization"
        ))

        # Create directories
        for dir_path in [self.config["output_dir"], self.config["temp_dir"]]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.audio_loader = AudioLoader(self.config)
        self.separator = MVSEPSeparator(self.config)
        self.analyzer = AudioAnalyzer(self.config)
        self.visualizer = AudioVisualizer(self.config)

    def process_audio(self, input_path: Path, model: str = "ensemble_extra") -> Dict[str, Any]:
        """
        Process audio file through separation pipeline.

        Args:
            input_path: Path to input audio file
            model: MVSEP model to use

        Returns:
            Processing results
        """
        start_time = time.time()
        results = {
            "input": str(input_path),
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
            "metrics": {}
        }

        try:
            # Load and analyze audio
            self._print_stage("Loading Audio", "📂")
            audio_data = self.audio_loader.load(input_path)

            # Quick analysis
            if self.config.get("analyze", True):
                analysis = self.analyzer.analyze(
                    audio_data["waveform"],
                    audio_data["sample_rate"]
                )
                results["stages"]["analysis"] = analysis

            # Source separation
            self._print_stage("Source Separation", "🎛️")
            self.console.print(f"[yellow]Using MVSEP model: {model}[/yellow]")

            # Perform separation
            separated = self.separator.separate(
                input_path,
                model=model,
                progress_callback=self._progress_callback
            )

            # Save separated stems
            output_dir = self.config["output_dir"] / "separated"
            output_dir.mkdir(exist_ok=True)

            stem_paths = {}
            for stem_name, stem_audio in separated["stems"].items():
                stem_path = output_dir / f"{audio_data['filename']}_{stem_name}.wav"
                torchaudio.save(stem_path, stem_audio, audio_data["sample_rate"])
                stem_paths[stem_name] = str(stem_path)

                # Create visualization if enabled
                if self.config["visualization"]:
                    viz_path = output_dir / f"{audio_data['filename']}_{stem_name}_viz.png"
                    self.visualizer.create_spectrogram(
                        stem_audio,
                        audio_data["sample_rate"],
                        viz_path,
                        title=f"{stem_name.capitalize()} Stem"
                    )

            # Store results
            results["stages"]["separation"] = {
                "model": separated["model"],
                "sep_type": separated.get("sep_type"),
                "stems": stem_paths,
                "metrics": separated.get("metrics", {}),
                "processing_time": separated.get("processing_time", 0)
            }

            # Calculate total time
            results["total_time"] = time.time() - start_time

            # Save results
            self._save_results(results)

            # Display summary
            self._display_summary(results)

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            results["error"] = str(e)
            raise

        return results

    def _print_stage(self, stage: str, emoji: str):
        """Print a stage header."""
        self.console.print(f"\n{emoji} [bold cyan]{stage}[/bold cyan]")

    def _progress_callback(self, progress: float, message: str):
        """Callback for progress updates."""
        self.console.print(f"[dim]{message}: {progress:.1%}[/dim]")

    def _save_results(self, results: Dict):
        """Save processing results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.config["output_dir"] / f"results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.console.print(f"[green]Results saved to {results_file}[/green]")

    def _display_summary(self, results: Dict):
        """Display processing summary."""
        table = Table(title="Processing Summary", show_header=True)
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")

        # Analysis stage
        if "analysis" in results["stages"]:
            analysis = results["stages"]["analysis"]
            details = f"Duration: {analysis.get('duration', 0):.1f}s"
            table.add_row("Analysis", "✓ Complete", details)

        # Separation stage
        if "separation" in results["stages"]:
            sep = results["stages"]["separation"]
            num_stems = len(sep.get("stems", {}))
            time_taken = sep.get("processing_time", 0)
            details = f"{num_stems} stems in {time_taken:.1f}s"
            table.add_row("Separation", "✓ Complete", details)

        # Total
        table.add_row(
            "Total",
            "Complete",
            f"{results.get('total_time', 0):.1f}s"
        )

        self.console.print("\n")
        self.console.print(table)

        # Show stem files
        if "separation" in results["stages"]:
            stems = results["stages"]["separation"].get("stems", {})
            if stems:
                self.console.print("\n[bold]Generated Stem Files:[/bold]")
                for stem_name, path in stems.items():
                    self.console.print(f"  • {stem_name}: [cyan]{Path(path).name}[/cyan]")


def test_api_connection():
    """Test MVSEP API connection and list available models."""
    console.print(Panel.fit(
        "[bold cyan]Testing MVSEP API Connection[/bold cyan]",
        title="🔌 API Test"
    ))

    try:
        config = {"mvsep": {"api_timeout": 60}}
        separator = MVSEPSeparator(config)

        algorithms = separator.get_available_algorithms()

        if algorithms:
            console.print("[green]✓ API connection successful![/green]\n")
            console.print("[bold]Available MVSEP Models:[/bold]")

            table = Table(show_header=True)
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="yellow")
            table.add_column("Description", style="white")

            for algo in algorithms[:10]:  # Show first 10
                table.add_row(
                    algo['id'],
                    algo['name'],
                    algo.get('description', '')[:50] + "..."
                )

            console.print(table)
        else:
            console.print("[red]Failed to retrieve algorithms[/red]")

    except Exception as e:
        console.print(f"[red]API test failed: {e}[/red]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Audio Source Separation using MVSEP API"
    )

    # Commands
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Separate command
    sep_parser = subparsers.add_parser('separate', help='Separate audio file')
    sep_parser.add_argument("--input", "-i", type=Path, required=True,
                            help="Input audio file")
    sep_parser.add_argument("--model", "-m", default="ensemble_extra",
                            choices=["ensemble_extra", "ensemble", "fast", "mdx", "reverb", "denoise"],
                            help="MVSEP model to use")
    sep_parser.add_argument("--output", "-o", type=Path, default=Path("output"),
                            help="Output directory")
    sep_parser.add_argument("--visualize", "-v", action="store_true",
                            help="Generate visualizations")

    # Test command
    test_parser = subparsers.add_parser('test', help='Test API connection')

    # List command
    list_parser = subparsers.add_parser('list', help='List available models')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'test' or args.command == 'list':
        test_api_connection()
        return

    if args.command == 'separate':
        # Print banner
        console.print(Panel.fit(
            "[bold magenta]MVSEP Audio Source Separation[/bold magenta]\n"
            f"📂 Input: {args.input}\n"
            f"🎛️ Model: {args.model}\n"
            f"📁 Output: {args.output}",
            title="🎵 Audio Separation",
            border_style="magenta"
        ))

        # Check API key
        if not os.environ.get("MVSEP_API_KEY"):
            console.print("\n[bold red]ERROR: MVSEP_API_KEY not set![/bold red]")
            console.print("Please add to your .env file:")
            console.print("MVSEP_API_KEY=your-api-key-here")
            sys.exit(1)

        # Configure pipeline
        config = {
            "output_dir": args.output,
            "temp_dir": Path("temp"),
            "visualization": args.visualize,
            "mvsep": {
                "api_timeout": 600,
                "quality": "high"
            }
        }

        # Run pipeline
        pipeline = AudioSeparationPipeline(config)
        results = pipeline.process_audio(args.input, args.model)

        console.print("\n[bold green]✨ Separation complete![/bold green]")
        console.print(f"Check the output directory: {args.output}")


if __name__ == "__main__":
    main()