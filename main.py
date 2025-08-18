#!/usr/bin/env python3
"""
Educational Audio Processing Pipeline
=====================================
Advanced audio manipulation using Meta's AI models for learning purposes.
Demonstrates source separation, style transfer, and audio enhancement.

Author: Educational AI Audio Lab
Version: 1.0.0
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time
from datetime import datetime

import numpy as np
import torch
import torchaudio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Import pipeline modules
from modules.audio_loader import AudioLoader
from modules.demucs_separator import DemucsSeparator
from modules.audiocraft_processor import AudioCraftProcessor
from modules.ddsp_transfer import DDSPStyleTransfer
from modules.audio_analyzer import AudioAnalyzer
from modules.deepseek_assistant import DeepSeekAssistant
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Configure rich console for beautiful output
console = Console()

# Configure logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


class EducationalAudioPipeline:
    """
    Main pipeline orchestrator for educational audio processing.
    Coordinates various AI models and provides learning insights.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the educational audio processing pipeline."""
        self.config = config or self._default_config()
        self.console = console

        # Initialize components
        self._initialize_components()

        # Track processing history for educational purposes
        self.processing_history = []
        self.learning_insights = []

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration for the pipeline."""
        return {
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "sample_rate": 44100,
            "output_dir": Path("output"),
            "temp_dir": Path("temp"),
            "models_dir": Path("models"),
            "educational_mode": False,
            "verbose": True,
            "save_intermediates": True,
            "visualization": False,
            "analysis": True,
            "demucs": {
                "model": "htdemucs_ft",  # Faster model by default
                "split": True,
                "two_stems": None,
                "mp3": True,
                "mp3_rate": 320,
                "float32": False,
                "int24": False,
                "shifts": 6,  # Balanced quality/speed
                "overlap": 0.25,  # Reduced for speed
            },
            "audiocraft": {
                "model": "musicgen-medium",
                "use_sampling": True,
                "top_k": 250,
                "top_p": 0.0,
                "temperature": 1.0,
                "duration": 8.0,
            },
            "ddsp": {
                "checkpoint": "violin",
                "pitch_shift": 0,
                "loudness_shift": 0,
                "f0_octave_shift": 0,
            },
            "deepseek": {
                "api_key": None,
                "model": "deepseek-chat",
                "temperature": 0.7,
                "max_tokens": 2048,
            }
        }

    def _initialize_components(self):
        """Initialize all pipeline components with progress tracking."""
        self.console.print(Panel.fit(
            "[bold cyan]Initializing Educational Audio Pipeline[/bold cyan]\n"
            "Loading AI models and preparing learning environment...",
            title="🎵 Pipeline Initialization"
        ))

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
        ) as progress:
            # Create directories
            task1 = progress.add_task("Creating directories...", total=100)
            for i, dir_path in enumerate([self.config["output_dir"],
                                          self.config["temp_dir"],
                                          self.config["models_dir"]]):
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                progress.update(task1, advance=33)

            # Initialize components with progress tracking
            task2 = progress.add_task("Loading audio utilities...", total=100)
            self.audio_loader = AudioLoader(self.config)
            progress.update(task2, advance=100)

            task3 = progress.add_task("Initializing Demucs v4 (downloading if needed)...", total=100)
            self.separator = DemucsSeparator(self.config)
            progress.update(task3, advance=100)

            task4 = progress.add_task("Loading AudioCraft suite (downloading if needed)...", total=100)
            try:
                self.audiocraft = AudioCraftProcessor(self.config)
                progress.update(task4, advance=100)
            except Exception as e:
                logger.warning(f"AudioCraft loading failed: {e}")
                self.audiocraft = None
                progress.update(task4, advance=100)

            task5 = progress.add_task("Setting up DDSP...", total=100)
            self.style_transfer = DDSPStyleTransfer(self.config)
            progress.update(task5, advance=100)

            task6 = progress.add_task("Preparing analyzer...", total=100)
            self.analyzer = AudioAnalyzer(self.config)
            progress.update(task6, advance=100)

            if self.config["deepseek"]["api_key"]:
                task7 = progress.add_task("Connecting to DeepSeek API...", total=100)
                self.assistant = DeepSeekAssistant(self.config)
                progress.update(task7, advance=100)
            else:
                self.assistant = None

    def process_audio(self,
                      input_path: Path,
                      mode: str = "full",
                      interactive: bool = False) -> Dict[str, Any]:
        """
        Process audio file through the educational pipeline.

        Args:
            input_path: Path to input audio file
            mode: Processing mode (full, learning, separator, audiocraft, etc.)
            interactive: Enable interactive learning mode

        Returns:
            Processing results and educational insights
        """
        start_time = time.time()
        results = {
            "input": str(input_path),
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
            "metrics": {},
            "insights": []
        }

        try:
            # Stage 1: Load and analyze input audio
            self._print_stage("Loading Audio", "📂")
            audio_data = self._load_and_analyze(input_path, results)

            if interactive:
                self._interactive_checkpoint("Audio loaded", audio_data)

            # Stage 2: Source separation with Demucs v4
            if mode in ["full", "learning", "separator"]:
                self._print_stage("Source Separation (Demucs v4)", "🎛️")
                separated = self._separate_sources(audio_data, results, interactive)

                if interactive:
                    self._interactive_checkpoint("Sources separated", separated)

            # Stage 3: AudioCraft processing
            if mode in ["full", "audiocraft"]:
                self._print_stage("AudioCraft Processing", "🎨")
                processed = self._process_audiocraft(audio_data, results, interactive)

                if interactive:
                    self._interactive_checkpoint("AudioCraft applied", processed)

            # Stage 4: Style transfer with DDSP
            if mode in ["full", "style"]:
                self._print_stage("Style Transfer (DDSP)", "🎭")
                transferred = self._apply_style_transfer(audio_data, results, interactive)

                if interactive:
                    self._interactive_checkpoint("Style transferred", transferred)

            # Calculate total processing time
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

    def _load_and_analyze(self, input_path: Path, results: Dict) -> Dict:
        """Load audio and perform initial analysis."""
        # Load audio
        audio_data = self.audio_loader.load(input_path)

        # Analyze audio features
        analysis = self.analyzer.analyze(audio_data["waveform"], audio_data["sample_rate"])

        # Store in results
        results["stages"]["loading"] = {
            "duration": audio_data["duration"],
            "sample_rate": audio_data["sample_rate"],
            "channels": audio_data["channels"],
            "format": audio_data["format"]
        }

        results["stages"]["analysis"] = analysis

        # Add educational insight
        if self.assistant:
            insight = self.assistant.explain_audio_features(analysis)
            results["insights"].append({
                "stage": "analysis",
                "content": insight
            })

        return audio_data

    def _separate_sources(self, audio_data: Dict, results: Dict, interactive: bool) -> Dict:
        """Perform source separation using Demucs v4 with progress tracking."""

        # Progress tracking for separation
        separation_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        )

        with separation_progress:
            task = separation_progress.add_task("Initializing separation...", total=100)

            def progress_callback(progress: float, message: str):
                separation_progress.update(task, completed=int(progress * 100), description=message)

            # Run separation with progress callback
            separated = self.separator.separate(
                audio_data["waveform"],
                audio_data["sample_rate"],
                progress_callback=progress_callback
            )

        # Save separated stems
        output_dir = self.config["output_dir"] / "separated"
        output_dir.mkdir(exist_ok=True)

        # Progress for saving stems
        save_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        )

        with save_progress:
            save_task = save_progress.add_task("Saving stems...", total=len(separated["stems"]))

            stem_paths = {}
            for i, (stem_name, stem_audio) in enumerate(separated["stems"].items()):
                stem_path = output_dir / f"{audio_data['filename']}_{stem_name}.wav"
                torchaudio.save(stem_path, stem_audio, audio_data["sample_rate"])
                stem_paths[stem_name] = str(stem_path)
                save_progress.update(save_task, advance=1, description=f"Saved {stem_name}")

        # Store results with detailed metrics
        results["stages"]["separation"] = {
            "model": self.config["demucs"]["model"],
            "stems": stem_paths,
            "metrics": separated.get("metrics", {}),
            "processing_time": separated.get("processing_time", 0),
            "snr_db": separated.get("metrics", {}).get("reconstruction_snr", 0),
            "quality_score": self._calculate_quality_score(separated.get("metrics", {}))
        }

        # Display separation quality metrics
        self._display_separation_metrics(separated.get("metrics", {}))

        # Educational insight about separation
        if self.assistant:
            insight = self.assistant.explain_source_separation(
                separated["metrics"],
                self.config["demucs"]["model"]
            )
            results["insights"].append({
                "stage": "separation",
                "content": insight
            })

        return separated

    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score based on metrics."""
        snr = metrics.get("reconstruction_snr", 0)
        num_stems = metrics.get("num_stems", 0)

        # Simple quality scoring based on SNR and stem count
        base_score = min(snr / 30.0, 1.0)  # Normalize SNR (30dB = excellent)
        stem_bonus = min(num_stems / 6.0, 1.0) * 0.1  # Bonus for more stems

        return min(base_score + stem_bonus, 1.0) * 100

    def _display_separation_metrics(self, metrics: Dict):
        """Display detailed separation quality metrics."""
        table = Table(title="Separation Quality Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Quality", style="yellow")

        snr = metrics.get("reconstruction_snr", 0)
        num_stems = metrics.get("num_stems", 0)

        # SNR Quality assessment
        if snr >= 25:
            snr_quality = "Excellent"
        elif snr >= 20:
            snr_quality = "Very Good"
        elif snr >= 15:
            snr_quality = "Good"
        elif snr >= 10:
            snr_quality = "Fair"
        else:
            snr_quality = "Poor"

        table.add_row("SNR (dB)", f"{snr:.2f}", snr_quality)
        table.add_row("Number of Stems", str(num_stems), "Standard" if num_stems >= 4 else "Limited")

        # Energy distribution
        for key, value in metrics.items():
            if key.endswith("_energy") and isinstance(value, (int, float)):
                stem_name = key.replace("_energy", "")
                table.add_row(f"{stem_name.title()} Energy", f"{value:.4f}", "-")

        self.console.print("\n")
        self.console.print(table)

    def _process_audiocraft(self, audio_data: Dict, results: Dict, interactive: bool) -> Dict:
        """Process audio using AudioCraft suite."""
        # Apply AudioCraft processing
        processed = self.audiocraft.process(
            audio_data["waveform"],
            audio_data["sample_rate"],
            mode="enhance"
        )

        # Save processed audio
        output_path = self.config["output_dir"] / f"{audio_data['filename']}_audiocraft.wav"
        torchaudio.save(output_path, processed["audio"], processed["sample_rate"])

        # Store results
        results["stages"]["audiocraft"] = {
            "model": self.config["audiocraft"]["model"],
            "output": str(output_path),
            "parameters": self.config["audiocraft"],
            "metrics": processed.get("metrics", {})
        }

        # Educational insight
        if self.assistant:
            insight = self.assistant.explain_audiocraft_processing(
                self.config["audiocraft"],
                processed.get("metrics", {})
            )
            results["insights"].append({
                "stage": "audiocraft",
                "content": insight
            })

        return processed

    def _apply_style_transfer(self, audio_data: Dict, results: Dict, interactive: bool) -> Dict:
        """Apply DDSP style transfer."""
        # Apply style transfer
        transferred = self.style_transfer.transfer(
            audio_data["waveform"],
            audio_data["sample_rate"],
            target_instrument=self.config["ddsp"]["checkpoint"]
        )

        # Save transferred audio
        output_path = self.config["output_dir"] / f"{audio_data['filename']}_ddsp.wav"
        torchaudio.save(output_path, transferred["audio"], transferred["sample_rate"])

        # Store results
        results["stages"]["style_transfer"] = {
            "instrument": self.config["ddsp"]["checkpoint"],
            "output": str(output_path),
            "parameters": self.config["ddsp"],
            "metrics": transferred.get("metrics", {})
        }

        # Educational insight
        if self.assistant:
            insight = self.assistant.explain_style_transfer(
                self.config["ddsp"],
                transferred.get("metrics", {})
            )
            results["insights"].append({
                "stage": "style_transfer",
                "content": insight
            })

        return transferred

    def _interactive_checkpoint(self, stage: str, data: Dict):
        """Interactive checkpoint for learning mode."""
        self.console.print(Panel.fit(
            f"[bold yellow]Interactive Checkpoint: {stage}[/bold yellow]\n"
            "Press Enter to continue, 'a' to analyze, or 'q' to quit.",
            title="🎓 Learning Mode"
        ))

        choice = input("Your choice: ").lower().strip()

        if choice == 'q':
            self.console.print("[red]Exiting pipeline...[/red]")
            sys.exit(0)
        elif choice == 'a':
            self._display_analysis(data)
            self._interactive_checkpoint(stage, data)

    def _display_analysis(self, data: Dict):
        """Display detailed analysis of current data."""
        table = Table(title="Audio Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in data.items():
            if isinstance(value, (int, float)):
                table.add_row(key, f"{value:.4f}")
            elif isinstance(value, str):
                table.add_row(key, value[:50] + "..." if len(value) > 50 else value)

        self.console.print(table)

    def _progress_callback(self, progress: float, message: str):
        """Callback for progress updates during processing."""
        self.console.print(f"[dim]{message}: {progress:.1%}[/dim]")

    def _print_stage(self, stage: str, emoji: str):
        """Print a stage header."""
        self.console.print(f"\n{emoji} [bold cyan]{stage}[/bold cyan]")

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
        table.add_column("Time (s)", style="yellow")
        table.add_column("Quality", style="magenta")

        for stage, data in results["stages"].items():
            time_taken = data.get("processing_time", "-")
            if isinstance(time_taken, float):
                time_taken = f"{time_taken:.2f}"

            quality = "-"
            if stage == "separation":
                quality = f"SNR: {data.get('snr_db', 0):.1f}dB"

            table.add_row(stage.capitalize(), "✓ Complete", str(time_taken), quality)

        table.add_row("Total", "Complete", f"{results['total_time']:.2f}", "-")

        self.console.print("\n")
        self.console.print(table)

        if results.get("insights"):
            self.console.print("\n[bold]Educational Insights:[/bold]")
            for insight in results["insights"]:
                self.console.print(f"\n[cyan]{insight['stage'].capitalize()}:[/cyan]")
                self.console.print(insight['content'])


def main():
    """Main entry point for the educational audio pipeline."""
    parser = argparse.ArgumentParser(
        description="Educational Audio Processing Pipeline - Learn AI Audio Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output arguments
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input audio file (MP3, WAV)")
    parser.add_argument("--output", "-o", type=Path, default=Path("output"),
                        help="Output directory (default: output)")

    # Processing modes
    parser.add_argument("--mode", "-m",
                        choices=["full", "learning", "separator", "audiocraft", "style"],
                        default="learning",
                        help="Processing mode")

    # Model selection
    parser.add_argument("--separator", default="demucs",
                        choices=["demucs"],
                        help="Source separator model")
    parser.add_argument("--demucs-model", default="htdemucs_6s",
                        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s"],
                        help="Demucs model variant")
    parser.add_argument("--audiocraft-model", default="musicgen-medium",
                        help="AudioCraft model to use")

    # Quality settings
    parser.add_argument("--shifts", type=int, default=6,
                        help="Number of shifts for separation quality (default: 6, use 1-10)")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="Overlap factor for separation (default: 0.25, use 0.5 for higher quality)")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast settings (lower quality but faster)")
    parser.add_argument("--high-quality", action="store_true",
                        help="Use high quality settings (slower but better results)")

    # Features
    parser.add_argument("--interactive", action="store_true",
                        help="Enable interactive learning mode")
    parser.add_argument("--educational", action="store_true", default=True,
                        help="Enable educational features (default: True)")
    parser.add_argument("--analyze", action="store_true", default=True,
                        help="Perform detailed analysis")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    # Batch processing
    parser.add_argument("--batch", type=Path,
                        help="Batch process directory")

    # API configuration
    parser.add_argument("--deepseek-api-key",
                        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")

    args = parser.parse_args()

    # Print welcome banner
    console.print(Panel.fit(
        "[bold magenta]Educational Audio Processing Pipeline[/bold magenta]\n"
        "[dim]Learn advanced AI audio processing with Meta's latest models[/dim]\n\n"
        f"📂 Input: {args.input}\n"
        f"🎛️ Mode: {args.mode}\n"
        f"🤖 Models: Demucs v4 + AudioCraft + DDSP\n"
        f"📊 Quality Focus: SNR optimization",
        title="🎵 Welcome to Audio AI Learning Lab",
        border_style="magenta"
    ))

    # Prepare configuration with quality settings
    if args.fast:
        shifts = 3
        overlap = 0.25
        model = "htdemucs_ft"
    elif args.high_quality:
        shifts = 10
        overlap = 0.5
        model = "htdemucs_6s"
    else:
        shifts = args.shifts
        overlap = args.overlap
        model = args.demucs_model

    config = {
        "output_dir": args.output,
        "temp_dir": Path("temp"),
        "models_dir": Path("models"),
        "educational_mode": args.educational,
        "verbose": args.verbose,
        "visualization": False,
        "analysis": args.analyze,
        "demucs": {
            "model": model,
            "shifts": shifts,
            "overlap": overlap,
        },
        "audiocraft": {
            "model": args.audiocraft_model,
        },
        "deepseek": {
            "api_key": args.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY"),
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 2048,
        }
    }

    # Initialize pipeline
    pipeline = EducationalAudioPipeline(config)

    # Process audio
    if args.batch:
        # Batch processing
        audio_files = list(args.batch.glob("*.mp3")) + list(args.batch.glob("*.wav"))
        console.print(f"[cyan]Found {len(audio_files)} audio files for batch processing[/cyan]")

        results_list = []
        for audio_file in audio_files:
            console.print(f"\n[bold]Processing: {audio_file.name}[/bold]")
            results = pipeline.process_audio(audio_file, args.mode, args.interactive)
            results_list.append(results)
    else:
        # Single file processing
        results = pipeline.process_audio(args.input, args.mode, args.interactive)

    console.print("\n[bold green]✨ Processing complete! Check the output directory for results.[/bold green]")


if __name__ == "__main__":
    import os

    main()