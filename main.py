#!/usr/bin/env python3
"""
Educational Audio Processing Pipeline
=====================================
Advanced audio manipulation using AI models for learning purposes.
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
import matplotlib.pyplot as plt
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
from modules.mvsep_separator import MVSEPSeparator
from modules.audiocraft_processor import AudioCraftProcessor
from modules.ddsp_transfer import DDSPStyleTransfer
from modules.audio_analyzer import AudioAnalyzer
from modules.visualization import AudioVisualizer
from modules.deepseek_assistant import DeepSeekAssistant
from modules.educational_reporter import EducationalReporter
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
            "mvsep": {
                "default_model": "karaoke",
                "max_stems": 7,
                "use_llm_optimization": True,
                "api_timeout": 300,
                "quality": "high"  # high/medium/fast
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
                "api_key": None,  # Set via environment variable
                "model": "deepseek-chat",
                "temperature": 0.7,
                "max_tokens": 2048,
            }
        }

    def _initialize_components(self):
        """Initialize all pipeline components."""
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
            task = progress.add_task("Creating directories...", total=3)
            for dir_path in [self.config["output_dir"],
                             self.config["temp_dir"],
                             self.config["models_dir"]]:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                progress.advance(task)

            # Initialize components
            progress.add_task("Loading audio utilities...")
            self.audio_loader = AudioLoader(self.config)

            # Use MVSEP for separation
            progress.add_task("Initializing MVSEP API...")
            self.separator = MVSEPSeparator(self.config)

            progress.add_task("Loading AudioCraft suite...")
            self.audiocraft = AudioCraftProcessor(self.config)

            progress.add_task("Setting up DDSP...")
            self.style_transfer = DDSPStyleTransfer(self.config)

            progress.add_task("Preparing analyzer...")
            self.analyzer = AudioAnalyzer(self.config)

            progress.add_task("Configuring visualizer...")
            self.visualizer = AudioVisualizer(self.config)

            if self.config["deepseek"]["api_key"]:
                progress.add_task("Connecting to DeepSeek...")
                self.assistant = DeepSeekAssistant(self.config)
            else:
                self.assistant = None

            progress.add_task("Setting up reporter...")
            self.reporter = EducationalReporter(self.config)

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

            # Stage 2: Source separation using MVSEP
            if mode in ["full", "learning", "separator"]:
                self._print_stage("Source Separation (MVSEP)", "🎛️")
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

            # Stage 5: Generate educational report
            if self.config["educational_mode"]:
                self._print_stage("Educational Analysis", "📊")
                self._generate_educational_report(results)

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

        # Generate visualizations if enabled
        if self.config["visualization"]:
            viz_path = self.config["output_dir"] / f"{input_path.stem}_input_viz.png"
            self.visualizer.create_spectrogram(
                audio_data["waveform"],
                audio_data["sample_rate"],
                viz_path,
                title="Input Audio Spectrogram"
            )
            results["stages"]["loading"]["visualization"] = str(viz_path)

        # Add educational insight
        if self.assistant:
            insight = self.assistant.explain_audio_features(analysis)
            results["insights"].append({
                "stage": "analysis",
                "content": insight
            })

        return audio_data

    def _separate_sources(self, audio_data: Dict, results: Dict, interactive: bool) -> Dict:
        """Perform source separation using MVSEP."""
        # Step 1: Extract advanced features for intelligent model selection
        features = {}
        if hasattr(self.analyzer, '_extract_spectral_features'):
            features.update(self.analyzer._extract_spectral_features(
                audio_data["waveform"].mean(dim=0).numpy() if audio_data["waveform"].dim() > 1
                else audio_data["waveform"].numpy(),
                audio_data["sample_rate"]
            ))
        if hasattr(self.analyzer, '_extract_harmonic_features'):
            features.update(self.analyzer._extract_harmonic_features(
                audio_data["waveform"].mean(dim=0).numpy() if audio_data["waveform"].dim() > 1
                else audio_data["waveform"].numpy(),
                audio_data["sample_rate"]
            ))

        # Step 2: LLM classification for intelligent model selection
        classification = None
        model = self.config["mvsep"]["default_model"]
        stems = 4

        if self.assistant and features:
            classification = self.assistant.classify_instruments(features)
            if classification and "instruments" in classification:
                # Let MVSEP's internal logic select the best model
                self.console.print(
                    f"[cyan]Detected instruments: {', '.join([i['name'] for i in classification['instruments']])}[/cyan]")

        # Step 3: Run MVSEP separation
        input_path = Path(audio_data["filepath"])

        self.console.print(f"[yellow]Using MVSEP model: {model} with {stems} stems[/yellow]")

        separated = self.separator.separate(
            input_path,
            model=model,
            stems=stems,
            instrument_hints=classification,
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

            # Visualize each stem
            if self.config["visualization"]:
                viz_path = output_dir / f"{audio_data['filename']}_{stem_name}_viz.png"
                self.visualizer.create_spectrogram(
                    stem_audio,
                    audio_data["sample_rate"],
                    viz_path,
                    title=f"{stem_name.capitalize()} Stem Spectrogram"
                )

        # Store results
        results["stages"]["separation"] = {
            "model": separated["model"],
            "model_info": separated.get("model_info", {}),
            "stems": stem_paths,
            "metrics": separated.get("metrics", {}),
            "processing_time": separated.get("processing_time", 0),
            "llm_guided": separated.get("llm_guided", False)
        }

        # Educational insight about separation
        if self.assistant:
            insight = self.assistant.explain_source_separation(
                separated["metrics"],
                separated["model"]
            )
            results["insights"].append({
                "stage": "separation",
                "content": insight
            })

        return separated

    def _process_audiocraft(self, audio_data: Dict, results: Dict, interactive: bool) -> Dict:
        """Process audio using AudioCraft suite."""
        # Apply AudioCraft processing
        processed = self.audiocraft.process(
            audio_data["waveform"],
            audio_data["sample_rate"],
            mode="enhance"  # or "generate" based on config
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

    def _generate_educational_report(self, results: Dict):
        """Generate comprehensive educational report."""
        report = self.reporter.generate_report(results)

        # Save report
        report_path = self.config["output_dir"] / "educational_report.html"
        report_path.write_text(report["html"])

        # Also save as JSON for programmatic access
        json_path = self.config["output_dir"] / "processing_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        results["report"] = {
            "html": str(report_path),
            "json": str(json_path)
        }

        self.console.print(f"[green]Educational report saved to {report_path}[/green]")

    def _interactive_checkpoint(self, stage: str, data: Dict):
        """Interactive checkpoint for learning mode."""
        self.console.print(Panel.fit(
            f"[bold yellow]Interactive Checkpoint: {stage}[/bold yellow]\n"
            "Press Enter to continue, 'a' to analyze, 'v' to visualize, or 'q' to quit.",
            title="🎓 Learning Mode"
        ))

        choice = input("Your choice: ").lower().strip()

        if choice == 'q':
            self.console.print("[red]Exiting pipeline...[/red]")
            sys.exit(0)
        elif choice == 'a':
            self._display_analysis(data)
            self._interactive_checkpoint(stage, data)  # Re-prompt
        elif choice == 'v':
            self._display_visualization(data)
            self._interactive_checkpoint(stage, data)  # Re-prompt

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

    def _display_visualization(self, data: Dict):
        """Display visualization of current data."""
        if "audio" in data or "waveform" in data:
            audio = data.get("audio", data.get("waveform"))
            sr = data.get("sample_rate", self.config["sample_rate"])

            # Create temporary visualization
            viz_path = self.config["temp_dir"] / "interactive_viz.png"
            self.visualizer.create_combined_plot(audio, sr, viz_path)

            self.console.print(f"[green]Visualization saved to {viz_path}[/green]")

            # Try to open in default viewer (macOS)
            import subprocess
            subprocess.run(["open", str(viz_path)])

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

        for stage, data in results["stages"].items():
            time_taken = data.get("processing_time", "-")
            if isinstance(time_taken, float):
                time_taken = f"{time_taken:.2f}"
            table.add_row(stage.capitalize(), "✓ Complete", str(time_taken))

        table.add_row("Total", "Complete", f"{results['total_time']:.2f}")

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

    # MVSEP Model selection
    parser.add_argument("--mvsep-model", default="ensemble",
                        choices=["ensemble", "ensemble_extra", "fast", "ultra_fast",
                                 "vocal_instrumental", "karaoke", "drums_focus",
                                 "piano", "guitar", "strings"],
                        help="MVSEP model variant")
    parser.add_argument("--mvsep-quality", default="high",
                        choices=["high", "medium", "fast"],
                        help="MVSEP processing quality")
    parser.add_argument("--mvsep-stems", type=int, default=4,
                        help="Number of stems to separate (2-7)")

    # AudioCraft settings
    parser.add_argument("--audiocraft-model", default="musicgen-medium",
                        help="AudioCraft model to use")

    # Features
    parser.add_argument("--interactive", action="store_true",
                        help="Enable interactive learning mode")
    parser.add_argument("--educational", action="store_true", default=True,
                        help="Enable educational features (default: True)")
    parser.add_argument("--analyze", action="store_true", default=True,
                        help="Perform detailed analysis")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Generate visualizations")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    # Batch processing
    parser.add_argument("--batch", type=Path,
                        help="Batch process directory")
    parser.add_argument("--compare-models", action="store_true",
                        help="Compare different model outputs")

    # API configuration
    parser.add_argument("--deepseek-api-key",
                        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")

    args = parser.parse_args()

    # Print welcome banner
    console.print(Panel.fit(
        "[bold magenta]Educational Audio Processing Pipeline[/bold magenta]\n"
        "[dim]Learn advanced AI audio processing with state-of-the-art models[/dim]\n\n"
        f"📂 Input: {args.input}\n"
        f"🎛️ Mode: {args.mode}\n"
        f"🤖 Models: MVSEP + AudioCraft + DDSP",
        title="🎵 Welcome to Audio AI Learning Lab",
        border_style="magenta"
    ))

    # Prepare configuration
    config = {
        "output_dir": args.output,
        "temp_dir": Path("temp"),
        "models_dir": Path("models"),
        "educational_mode": args.educational,
        "verbose": args.verbose,
        "visualization": args.visualize,
        "analysis": args.analyze,
        "mvsep": {
            "default_model": args.mvsep_model,
            "quality": args.mvsep_quality,
            "max_stems": args.mvsep_stems,
            "use_llm_optimization": True,
            "api_timeout": 300,
        },
        "audiocraft": {
            "model": args.audiocraft_model,
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

        # Generate comparison report if requested
        if args.compare_models:
            pipeline.reporter.generate_comparison_report(results_list)
    else:
        # Single file processing
        results = pipeline.process_audio(args.input, args.mode, args.interactive)

    console.print("\n[bold green]✨ Processing complete! Check the output directory for results.[/bold green]")


if __name__ == "__main__":
    main()