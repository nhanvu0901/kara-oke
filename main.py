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
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
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
from modules.stem_analyzer import StemAnalyzer
from modules.stem_transformer import StemTransformer
from modules.interactive_selector import InteractiveSelector
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
        self.stem_analyzer = StemAnalyzer(self.config["sample_rate"])
        self.stem_transformer = StemTransformer(self.config)
        self.interactive_selector = InteractiveSelector()

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

    def _transform_stems_enhanced(self, separated: Dict, results: Dict, interactive: bool) -> Dict:
        """Enhanced transform that works with pre-loaded stems."""

        from modules.stem_transformer import StemTransformer

        # Initialize transformer
        transformer = StemTransformer(self.config, self.audiocraft)

        # Use existing analyses if available, otherwise analyze now
        if "stem_analyses" in separated:
            stem_analyses = separated["stem_analyses"]
        else:
            stem_analyses = {}
            for stem_name, stem_audio in separated["stems"].items():
                self.console.print(f"\n[cyan]Analyzing {stem_name}...[/cyan]")
                analysis = self.stem_analyzer.analyze_stem(stem_audio, stem_name)
                analysis["sample_rate"] = separated.get("sample_rate", 44100)
                stem_analyses[stem_name] = analysis

        # Display analysis summary
        self._display_analysis_summary(stem_analyses)

        # Get transformations
        transformations = {}

        if interactive:
            # Interactive selection
            self.console.print("\n[bold cyan]Interactive Transformation Selection[/bold cyan]")
            for stem_name, analysis in stem_analyses.items():
                selection = self.interactive_selector.display_stem_options(analysis)
                if selection != "skip":
                    transformations[stem_name] = selection
        else:
            # Auto-select best transformations
            self.console.print("\n[cyan]Auto-selecting optimal transformations...[/cyan]")
            transformations = self._auto_select_transformations(stem_analyses)

        if transformations:
            # Show summary
            if interactive:
                if not self.interactive_selector.display_transformation_summary(transformations):
                    self.console.print("[yellow]Transformation cancelled[/yellow]")
                    return separated

            # Apply transformations with progress tracking
            self.console.print("\n[bold green]Applying Transformations[/bold green]")

            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=self.console
            ) as progress:

                task = progress.add_task("Transforming stems...", total=len(transformations))

                # Batch transform
                transformed_stems = transformer.batch_transform(
                    separated["stems"],
                    stem_analyses,
                    transformations,
                    parallel=self.config.get("stem_transformation", {}).get("parallel_processing", True)
                )

                progress.update(task, completed=len(transformations))

            # Save transformed stems
            self._save_transformed_outputs(transformed_stems, separated["stems"])

            # Quality report
            self._display_quality_report(transformed_stems, transformations)

            # Store results
            results["stages"]["transformation"] = {
                "analyses": {k: {
                    "instrument_type": v["instrument_type"],
                    "key": v.get("key", "Unknown"),
                    "energy": v.get("energy", 0)
                } for k, v in stem_analyses.items()},
                "transformations": transformations,
                "quality_scores": {
                    k: v.get("quality_score", 0)
                    for k, v in transformed_stems.items()
                },
                "harmonic_compatibility": {
                    k: v.get("harmonic_compatibility", 0)
                    for k, v in transformed_stems.items()
                }
            }

            return transformed_stems
        else:
            self.console.print("[yellow]No transformations selected[/yellow]")
            return separated

    def _display_analysis_summary(self, stem_analyses: Dict):
        """Display comprehensive analysis summary."""

        table = Table(title="🎵 Stem Analysis Summary", show_header=True)
        table.add_column("Stem", style="cyan", width=15)
        table.add_column("Type", style="yellow", width=12)
        table.add_column("Key", style="magenta", width=8)
        table.add_column("Energy", style="green", width=10)
        table.add_column("Tempo Rel.", style="blue", width=12)
        table.add_column("Harmonic", style="red", width=10)

        for name, analysis in stem_analyses.items():
            tempo_rel = analysis.get("tempo_relevance", 0)
            harmonic = analysis.get("harmonic_content", {}).get("harmonic_ratio", 0)

            table.add_row(
                name,
                analysis.get("instrument_type", "unknown"),
                analysis.get("key", "?"),
                f"{analysis.get('energy', 0):.3f}",
                f"{tempo_rel:.0%}",
                f"{harmonic:.0%}"
            )

        self.console.print(table)

    def _save_transformed_outputs(self, transformed_stems: Dict, original_stems: Dict):
        """Save all transformation outputs with comparison."""

        output_dir = self.config["output_dir"] / "transformed"
        output_dir.mkdir(exist_ok=True)

        # Save individual transformed stems
        for stem_name, transform_data in transformed_stems.items():
            if "audio" in transform_data:
                # Transformed version
                trans_path = output_dir / f"{stem_name}_{transform_data.get('transformation', 'transformed')}.wav"
                torchaudio.save(
                    trans_path,
                    transform_data["audio"],
                    self.config.get("sample_rate", 44100)
                )

                # Also save original for comparison if requested
                if self.config.get("save_comparisons", True):
                    orig_path = output_dir / f"{stem_name}_original.wav"
                    if stem_name in original_stems:
                        torchaudio.save(
                            orig_path,
                            original_stems[stem_name],
                            self.config.get("sample_rate", 44100)
                        )

        # Create mixed versions
        self._create_mixed_output(transformed_stems, output_dir)

        # Create comparison mix (original vs transformed)
        if self.config.get("save_comparisons", True):
            self._create_comparison_mix(original_stems, transformed_stems, output_dir)

        self.console.print(f"\n[green]✅ All outputs saved to {output_dir}[/green]")

    def _display_quality_report(self, transformed_stems: Dict, transformations: Dict):
        """Display detailed quality report of transformations."""

        table = Table(title="🎯 Transformation Quality Report", show_header=True)
        table.add_column("Stem", style="cyan")
        table.add_column("Transformation", style="yellow")
        table.add_column("Quality", style="green")
        table.add_column("Harmonic", style="magenta")
        table.add_column("Time", style="blue")

        total_quality = 0
        num_transforms = 0

        for stem_name, transform_data in transformed_stems.items():
            if stem_name in transformations:
                quality = transform_data.get("quality_score", 0)
                harmonic = transform_data.get("harmonic_compatibility", 0)
                time = transform_data.get("processing_time", 0)

                # Color code quality
                if quality >= 0.8:
                    quality_str = f"[green]{quality:.0%}[/green]"
                elif quality >= 0.6:
                    quality_str = f"[yellow]{quality:.0%}[/yellow]"
                else:
                    quality_str = f"[red]{quality:.0%}[/red]"

                table.add_row(
                    stem_name,
                    transformations[stem_name],
                    quality_str,
                    f"{harmonic:.0%}",
                    f"{time:.1f}s"
                )

                total_quality += quality
                num_transforms += 1

        self.console.print(table)

        if num_transforms > 0:
            avg_quality = total_quality / num_transforms
            self.console.print(f"\n[bold]Average Quality Score: {avg_quality:.0%}[/bold]")

    def _create_comparison_mix(self, original_stems: Dict, transformed_stems: Dict, output_dir: Path):
        """Create side-by-side comparison mix."""

        # Implementation for A/B comparison
        # Left channel: original mix
        # Right channel: transformed mix
        # Or create two separate files for easier comparison
        pass
    def process_audio(self,
                      input_path: Path,
                      mode: str = "full",
                      interactive: bool = False,
                      results_file: Optional[Path] = None) -> Dict[str, Any]:  # ADD results_file parameter
        """
        Process audio file through the educational pipeline.

        Args:
            input_path: Path to input audio file or results JSON for transform-only mode
            mode: Processing mode (full, learning, separator, audiocraft, transform, transform-only)
            interactive: Enable interactive learning mode
            results_file: Path to existing results JSON (for transform-only mode)

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
            # Check if transform-only mode
            if mode == "transform-only":
                if not results_file:
                    # If no results file specified, try to find the latest one
                    results_files = list(self.config["output_dir"].glob("results_*.json"))
                    if results_files:
                        results_file = max(results_files, key=lambda p: p.stat().st_mtime)
                        self.console.print(f"[yellow]Using latest results file: {results_file.name}[/yellow]")
                    else:
                        raise ValueError("No results file found. Please run separation first or specify --results-file")

                # Load existing stems
                separated = self._load_existing_stems(results_file, results)

                # Jump directly to transformation
                self._print_stage("Stem Transformation", "🎭")
                transformed = self._transform_stems_enhanced(separated, results, interactive)

            else:
                # Normal processing flow
                # Stage 1: Load and analyze input audio
                self._print_stage("Loading Audio", "📂")
                audio_data = self._load_and_analyze(input_path, results)

                if interactive:
                    self._interactive_checkpoint("Audio loaded", audio_data)

                # Stage 2: Source separation with Demucs v4
                if mode in ["full", "learning", "separator", "transform"]:
                    self._print_stage("Source Separation (Demucs v4)", "🎛️")
                    separated = self._separate_sources(audio_data, results, interactive)

                    if interactive:
                        self._interactive_checkpoint("Sources separated", separated)

                    # Add stem analysis for transformation
                    if mode in ["full", "transform"]:
                        separated["stem_analyses"] = {}
                        for stem_name, stem_audio in separated["stems"].items():
                            analysis = self.stem_analyzer.analyze_stem(stem_audio, stem_name)
                            analysis["sample_rate"] = audio_data["sample_rate"]
                            separated["stem_analyses"][stem_name] = analysis

                # Stage 3: Transformation (if requested)
                if mode in ["full", "transform"]:
                    self._print_stage("Stem Transformation", "🎭")
                    transformed = self._transform_stems_enhanced(separated, results, interactive)

                    if interactive:
                        self._interactive_checkpoint("Stems transformed", transformed)

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

    # In main.py, add these methods to EducationalAudioPipeline class:
    def _load_existing_stems(self, results_file: Path, results: Dict) -> Dict:
        """Load existing separated stems from previous run."""

        self._print_stage("Loading Existing Stems", "📁")

        import json

        # Load results JSON
        with open(results_file, 'r') as f:
            previous_results = json.load(f)

        # Extract stem information
        if "stages" not in previous_results or "separation" not in previous_results["stages"]:
            raise ValueError("Results file doesn't contain separation data")

        separation_data = previous_results["stages"]["separation"]
        stems_paths = separation_data.get("stems", {})

        # Load audio files
        stems = {}
        stem_analyses = {}

        self.console.print(f"[cyan]Loading {len(stems_paths)} stems...[/cyan]")

        for stem_name, stem_path in stems_paths.items():
            stem_file = Path(stem_path)

            if not stem_file.exists():
                # Try relative to output directory
                stem_file = self.config["output_dir"] / "separated" / stem_file.name

            if stem_file.exists():
                # Load audio
                waveform, sample_rate = torchaudio.load(stem_file)
                stems[stem_name] = waveform

                self.console.print(f"  ✅ Loaded: {stem_name} ({stem_file.name})")

                # Analyze the stem
                analysis = self.stem_analyzer.analyze_stem(waveform, stem_name)
                analysis["sample_rate"] = sample_rate
                stem_analyses[stem_name] = analysis
            else:
                self.console.print(f"  ❌ Not found: {stem_file}")

        # Copy relevant metrics from previous results
        results["stages"]["loading"] = previous_results["stages"].get("loading", {})
        results["stages"]["analysis"] = previous_results["stages"].get("analysis", {})
        results["stages"]["separation"] = separation_data

        # Add insights from previous run
        if "insights" in previous_results:
            results["insights"] = previous_results["insights"]

        # Create separated dict format expected by transform pipeline
        separated = {
            "stems": stems,
            "source_names": list(stems.keys()),
            "metrics": separation_data.get("metrics", {}),
            "sample_rate": sample_rate,
            "stem_analyses": stem_analyses
        }

        self.console.print(f"\n[green]Successfully loaded {len(stems)} stems from previous run[/green]")

        # Display stem info
        table = Table(title="Loaded Stems", show_header=True)
        table.add_column("Stem", style="cyan")
        table.add_column("Detected Type", style="yellow")
        table.add_column("Energy", style="green")
        table.add_column("Key", style="magenta")

        for name, analysis in stem_analyses.items():
            table.add_row(
                name,
                analysis.get("instrument_type", "unknown"),
                f"{analysis.get('energy', 0):.4f}",
                analysis.get("key", "?")
            )

        self.console.print(table)

        return separated
    def _transform_stems(self, separated: Dict, results: Dict, interactive: bool) -> Dict:
        """Analyze and transform separated stems."""

        self._print_stage("Stem Analysis & Transformation", "🎭")

        # Initialize transformer with audiocraft
        from modules.stem_transformer import StemTransformer
        transformer = StemTransformer(self.config, self.audiocraft)

        # Analyze each stem
        stem_analyses = {}
        for stem_name, stem_audio in separated["stems"].items():
            self.console.print(f"\n[cyan]Analyzing {stem_name}...[/cyan]")
            analysis = self.stem_analyzer.analyze_stem(stem_audio, stem_name)
            stem_analyses[stem_name] = analysis
            analysis["sample_rate"] = separated.get("sample_rate", 44100)

        # Get transformations
        transformations = {}

        if interactive:
            # Interactive selection
            for stem_name, analysis in stem_analyses.items():
                selection = self.interactive_selector.display_stem_options(analysis)
                if selection != "skip":
                    transformations[stem_name] = selection
        else:
            # Auto-select best transformations
            transformations = self._auto_select_transformations(stem_analyses)

        if transformations:
            # Show summary
            if interactive:
                if not self.interactive_selector.display_transformation_summary(transformations):
                    self.console.print("[yellow]Transformation cancelled[/yellow]")
                    return separated

            # Apply transformations
            self.console.print("\n[cyan]Applying transformations...[/cyan]")
            transformed_stems = transformer.batch_transform(
                separated["stems"],
                stem_analyses,
                transformations,
                parallel=True
            )

            # Save transformed stems
            output_dir = self.config["output_dir"] / "transformed"
            output_dir.mkdir(exist_ok=True)

            for stem_name, transform_data in transformed_stems.items():
                if "audio" in transform_data:
                    output_path = output_dir / f"{stem_name}_{transform_data.get('transformation', 'transformed')}.wav"
                    torchaudio.save(
                        output_path,
                        transform_data["audio"],
                        self.config.get("sample_rate", 44100)
                    )
                    self.console.print(f"[green]Saved: {output_path.name}[/green]")

            # Create mixed version
            self._create_mixed_output(transformed_stems, output_dir)

            # Store results
            results["stages"]["transformation"] = {
                "analyses": {k: {
                    "instrument_type": v["instrument_type"],
                    "key": v.get("key", "Unknown"),
                    "energy": v.get("energy", 0)
                } for k, v in stem_analyses.items()},
                "transformations": transformations,
                "quality_scores": {
                    k: v.get("quality_score", 0)
                    for k, v in transformed_stems.items()
                }
            }

            return transformed_stems
        else:
            self.console.print("[yellow]No transformations selected[/yellow]")
            return separated

    def _auto_select_transformations(self, stem_analyses: Dict) -> Dict:
        """Automatically select best transformations based on compatibility."""
        transformations = {}

        for stem_name, analysis in stem_analyses.items():
            # Find best transformation based on compatibility
            best_transform = None
            best_score = 0.6  # Minimum threshold

            for transform_name, info in analysis["recommended_transformations"].items():
                if info.get("compatibility", 0) > best_score:
                    best_transform = transform_name
                    best_score = info["compatibility"]

            if best_transform:
                transformations[stem_name] = best_transform
                self.console.print(
                    f"[green]Auto-selected '{best_transform}' for {stem_name} "
                    f"(compatibility: {best_score:.0%})[/green]"
                )

        return transformations

    def _create_mixed_output(self, transformed_stems: Dict, output_dir: Path):
        """Create a mixed version of all transformed stems."""

        # Collect all audio
        all_audio = []
        max_length = 0

        for stem_data in transformed_stems.values():
            if "audio" in stem_data:
                audio = stem_data["audio"]
                all_audio.append(audio)
                max_length = max(max_length, audio.shape[-1])

        if all_audio:
            # Pad all to same length
            padded_audio = []
            for audio in all_audio:
                if audio.shape[-1] < max_length:
                    padding = max_length - audio.shape[-1]
                    audio = torch.nn.functional.pad(audio, (0, padding))
                padded_audio.append(audio)

            # Mix (simple sum and normalize)
            mixed = torch.sum(torch.stack(padded_audio), dim=0)
            mixed = mixed / len(padded_audio)  # Average

            # Normalize to prevent clipping
            max_val = torch.max(torch.abs(mixed))
            if max_val > 0.95:
                mixed = mixed * (0.95 / max_val)

            # Save mixed output
            mixed_path = output_dir / "mixed_transformed.wav"
            torchaudio.save(
                mixed_path,
                mixed,
                self.config.get("sample_rate", 44100)
            )

            self.console.print(f"[green]Created mixed output: {mixed_path.name}[/green]")

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
    parser.add_argument("--input", "-i", type=Path, required=False,  # Changed to not required
                        help="Input audio file (MP3, WAV) - not needed for transform-only mode")
    parser.add_argument("--output", "-o", type=Path, default=Path("output"),
                        help="Output directory (default: output)")

    # Processing modes
    parser.add_argument("--mode", "-m",
                        choices=["full", "learning", "separator", "audiocraft", "style", "transform", "transform-only"],
                        default="learning",
                        help="Processing mode (transform-only uses existing stems)")

    # Transform-specific arguments
    parser.add_argument("--results-file", type=Path,
                        help="Path to existing results JSON file (for transform-only mode)")
    parser.add_argument("--stems-dir", type=Path,
                        help="Directory containing separated stems (alternative to results-file)")
    parser.add_argument("--auto-transform", action="store_true",
                        help="Automatically select best transformations")
    parser.add_argument("--save-comparisons", action="store_true", default=True,
                        help="Save comparison files (original vs transformed)")

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

    # Validate input requirement
    if args.mode != "transform-only" and not args.input:
        parser.error("--input is required for all modes except transform-only")
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
        "sample_rate": 44100,
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
    if args.mode == "transform-only":
        # Special handling for transform-only mode
        if args.results_file and args.results_file.exists():
            # Use specified results file
            results = pipeline.process_audio(
                args.input,  # Can be dummy path
                mode="transform-only",
                interactive=args.interactive,
                results_file=args.results_file
            )
        elif args.stems_dir and args.stems_dir.exists():
            # Load stems directly from directory
            # (implement separate method for this if needed)
            results = pipeline.process_audio(
                args.stems_dir,
                mode="transform-only",
                interactive=args.interactive
            )
        else:
            # Try to find latest results automatically
            results = pipeline.process_audio(
                Path("dummy"),  # Placeholder
                mode="transform-only",
                interactive=args.interactive
            )
    else:
        # Normal processing
        results = pipeline.process_audio(
            args.input,
            args.mode,
            args.interactive
        )
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