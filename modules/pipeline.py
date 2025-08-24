"""
Separation Pipeline - Core processing logic
==========================================
Main pipeline orchestrator for educational audio source separation.
Focus on instrumental stems only (bass, drums, other).
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torchaudio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from .audio_loader import AudioLoader
from .demucs_separator import DemucsSeparator
from .audio_analyzer import AudioAnalyzer
from .deepseek_assistant import DeepSeekAssistant
from .visualization import AudioVisualizer

console = Console()


class SeparationPipeline:
    """
    Main pipeline orchestrator for educational audio source separation.
    Focuses on high-quality instrumental stem separation with detailed analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the separation pipeline."""
        self.config = config
        self.console = console
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all pipeline components."""
        self.console.print("🚀 Initializing pipeline components...")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            # Create directories
            task1 = progress.add_task("Creating directories...", total=100)
            self._create_directories()
            progress.update(task1, advance=100)

            # Initialize components
            task2 = progress.add_task("Loading audio utilities...", total=100)
            self.audio_loader = AudioLoader(self.config)
            progress.update(task2, advance=100)

            task3 = progress.add_task("Loading Demucs v4 (Instrumental)...", total=100)
            self.separator = DemucsSeparator(self.config)
            progress.update(task3, advance=100)

            task4 = progress.add_task("Preparing analyzer...", total=100)
            self.analyzer = AudioAnalyzer(self.config)
            progress.update(task4, advance=100)

            if self.config["visualization"]:
                task5 = progress.add_task("Setting up visualizer...", total=100)
                self.visualizer = AudioVisualizer(self.config)
                progress.update(task5, advance=100)
            else:
                self.visualizer = None

            if self.config["deepseek"]["api_key"]:
                task6 = progress.add_task("Connecting to DeepSeek API...", total=100)
                self.assistant = DeepSeekAssistant(self.config)
                progress.update(task6, advance=100)
            else:
                self.assistant = None

    def _create_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.config["output_dir"],
            self.config["output_dir"] / "separated",
            self.config["output_dir"] / "instrumentals",
            self.config["output_dir"] / "visualizations",
            self.config["output_dir"] / "reports",
            self.config["temp_dir"]
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def process_audio(self, input_path: Path, mode: str = "learning", interactive: bool = False) -> Dict[str, Any]:
        """Process a single audio file for instrumental separation."""
        start_time = time.time()

        results = {
            "input": str(input_path),
            "mode": mode,
            "type": "instrumental_separation",
            "timestamp": datetime.now().isoformat(),
            "stages": {},
            "insights": []
        }

        try:
            # Stage 1: Load and analyze
            self._print_stage("Loading & Analyzing Audio", "📂")
            audio_data = self._load_and_analyze(input_path, results)

            if interactive:
                self._interactive_checkpoint("Audio analyzed", audio_data)

            # Stage 2: Visualizations
            if self.config["visualization"] and mode in ["full", "learning", "analysis"]:
                self._print_stage("Creating Visualizations", "📊")
                self._create_visualizations(audio_data, results)

            # Stage 3: Instrumental Separation
            if mode in ["full", "learning", "separator"]:
                self._print_stage("Instrumental Source Separation", "🎛️")
                separated = self._separate_instrumental_sources(audio_data, results, interactive)

                if mode in ["full", "learning"]:
                    self._print_stage("Analyzing Instrumental Stems", "🔬")
                    self._analyze_instrumental_stems(separated, results)

            results["total_time"] = time.time() - start_time
            self._save_results(results)
            self._display_summary(results)

        except Exception as e:
            results["error"] = str(e)
            raise

        return results

    def process_batch(self, batch_dir: Path, mode: str = "learning", interactive: bool = False):
        """Process multiple audio files for instrumental separation."""
        audio_files = list(batch_dir.glob("*.mp3")) + list(batch_dir.glob("*.wav"))

        if not audio_files:
            console.print(f"[red]No audio files found in {batch_dir}[/red]")
            return

        console.print(f"[cyan]Processing {len(audio_files)} files for instrumental separation...[/cyan]")

        results_list = []
        for audio_file in audio_files:
            console.print(f"\n[bold]Processing: {audio_file.name}[/bold]")
            try:
                result = self.process_audio(audio_file, mode, False)  # No interactive for batch
                results_list.append(result)
            except Exception as e:
                console.print(f"[red]Failed: {e}[/red]")

        # Save batch summary
        batch_summary = {
            "total_files": len(audio_files),
            "successful": len(results_list),
            "type": "instrumental_batch_processing",
            "results": results_list,
            "timestamp": datetime.now().isoformat()
        }

        batch_file = self.config["output_dir"] / f"instrumental_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)

        console.print(f"[green]Batch complete! Summary: {batch_file}[/green]")

    def _load_and_analyze(self, input_path: Path, results: Dict) -> Dict:
        """Load and analyze audio."""
        audio_data = self.audio_loader.load(input_path)
        analysis = self.analyzer.analyze(audio_data["waveform"], audio_data["sample_rate"])

        results["stages"]["loading"] = {
            "duration": audio_data["duration"],
            "sample_rate": audio_data["sample_rate"],
            "channels": audio_data["channels"],
            "format": audio_data["format"]
        }
        results["stages"]["analysis"] = analysis

        self._display_audio_analysis(analysis)

        if self.assistant:
            insight = self.assistant.explain_audio_features(analysis)
            results["insights"].append({"stage": "analysis", "content": insight})

        return audio_data

    def _create_visualizations(self, audio_data: Dict, results: Dict):
        """Create audio visualizations."""
        if not self.visualizer:
            return

        viz_dir = self.config["output_dir"] / "visualizations"
        filename = audio_data["filename"]

        # Main analysis plot
        viz_path = viz_dir / f"{filename}_analysis.png"
        self.visualizer.create_combined_plot(
            audio_data["waveform"], audio_data["sample_rate"], viz_path
        )

        # Spectrogram
        spec_path = viz_dir / f"{filename}_spectrogram.png"
        self.visualizer.create_spectrogram(
            audio_data["waveform"], audio_data["sample_rate"], spec_path, f"{filename} - Original"
        )

        results["stages"]["visualization"] = {
            "analysis_plot": str(viz_path),
            "spectrogram": str(spec_path)
        }

    def _separate_instrumental_sources(self, audio_data: Dict, results: Dict, interactive: bool) -> Dict:
        """Perform instrumental source separation (no vocals)."""
        # Progress tracking
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Separating instrumental stems...", total=100)

            def progress_callback(prog: float, msg: str):
                progress.update(task, completed=int(prog * 100), description=msg)

            separated = self.separator.separate(
                audio_data["waveform"], audio_data["sample_rate"], progress_callback
            )

        # Save instrumental stems
        stem_paths = self._save_instrumental_stems(separated["stems"], audio_data)

        # Create stem visualizations
        if self.visualizer:
            self._create_instrumental_visualizations(separated["stems"], audio_data)

        results["stages"]["instrumental_separation"] = {
            "model": self.config["demucs"]["model"],
            "instrumental_stems": stem_paths,
            "metrics": separated.get("metrics", {}),
            "processing_time": separated.get("processing_time", 0),
            "note": "Instrumental stems only - vocals excluded"
        }

        self._display_instrumental_metrics(separated.get("metrics", {}))

        if self.assistant:
            insight = self.assistant.explain_source_separation(
                separated["metrics"], self.config["demucs"]["model"]
            )
            results["insights"].append({"stage": "instrumental_separation", "content": insight})

        return separated

    def _save_instrumental_stems(self, stems: Dict, audio_data: Dict) -> Dict:
        """Save separated instrumental stems to files."""
        output_dir = self.config["output_dir"] / "instrumentals"
        stem_paths = {}

        with Progress(SpinnerColumn(), TextColumn("Saving {task.description}...")) as progress:
            for stem_name, stem_audio in stems.items():
                task = progress.add_task(f"instrumental_{stem_name}", total=1)

                stem_path = output_dir / f"{audio_data['filename']}_{stem_name}.wav"
                torchaudio.save(stem_path, stem_audio, audio_data["sample_rate"])
                stem_paths[stem_name] = str(stem_path)

                progress.update(task, advance=1)

        return stem_paths

    def _create_instrumental_visualizations(self, stems: Dict, audio_data: Dict):
        """Create visualizations for each instrumental stem."""
        viz_dir = self.config["output_dir"] / "visualizations"
        filename = audio_data["filename"]

        for stem_name, stem_audio in stems.items():
            stem_viz_path = viz_dir / f"{filename}_instrumental_{stem_name}.png"
            self.visualizer.create_spectrogram(
                stem_audio, audio_data["sample_rate"], stem_viz_path,
                f"{filename} - Instrumental: {stem_name.title()}"
            )

    def _analyze_instrumental_stems(self, separated: Dict, results: Dict):
        """Analyze each separated instrumental stem."""
        stem_analysis = {}

        for stem_name, stem_audio in separated["stems"].items():
            analysis = self.analyzer.analyze(stem_audio, 44100)
            stem_analysis[stem_name] = analysis
            self._display_stem_analysis(f"Instrumental: {stem_name}", analysis)

        results["stages"]["instrumental_stem_analysis"] = stem_analysis

    def _display_audio_analysis(self, analysis: Dict):
        """Display audio analysis results."""
        table = Table(title="Audio Analysis", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Assessment", style="yellow")

        # Key metrics
        table.add_row("Duration", f"{analysis.get('duration', 0):.2f}s", "-")
        table.add_row("RMS Energy", f"{analysis.get('rms_energy', 0):.4f}",
                      self._assess_energy(analysis.get('rms_energy', 0)))
        table.add_row("Dynamic Range", f"{analysis.get('dynamic_range_db', 0):.1f} dB",
                      self._assess_dynamic_range(analysis.get('dynamic_range_db', 0)))

        if 'spectral_centroid_hz' in analysis:
            centroid = analysis['spectral_centroid_hz']
            table.add_row("Brightness", f"{centroid:.1f} Hz", self._assess_brightness(centroid))

        if 'harmonic_ratio' in analysis:
            harmonic = analysis['harmonic_ratio']
            table.add_row("Harmonic Content", f"{harmonic:.2%}",
                         "Rich" if harmonic > 0.6 else "Balanced" if harmonic > 0.3 else "Percussive")

        if 'separation_readiness_score' in analysis:
            score = analysis['separation_readiness_score']
            table.add_row("Separation Ready", f"{score:.2f}", self._assess_readiness(score))

        console.print(table)

        # Recommendation for instrumental separation
        console.print(f"\n[bold cyan]Instrumental Separation Recommendation:[/bold cyan]")
        if analysis.get('harmonic_percussive_ratio', 1) > 0.5:
            console.print("Good candidate for instrumental separation - clear harmonic content detected")
        else:
            console.print("Strong percussive content - drums should separate cleanly")

    def _display_stem_analysis(self, stem_name: str, analysis: Dict):
        """Display individual stem analysis."""
        table = Table(title=f"{stem_name}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Energy", f"{analysis.get('rms_energy', 0):.4f}")
        table.add_row("Peak", f"{analysis.get('peak_amplitude', 0):.4f}")
        table.add_row("Dynamic Range", f"{analysis.get('dynamic_range_db', 0):.1f} dB")

        if 'spectral_centroid_mean' in analysis:
            table.add_row("Brightness", f"{analysis.get('spectral_centroid_mean', 0):.1f} Hz")

        console.print(table)

    def _display_instrumental_metrics(self, metrics: Dict):
        """Display instrumental separation quality metrics."""
        table = Table(title="Instrumental Separation Quality", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Grade", style="yellow")

        coverage = metrics.get("instrumental_coverage_ratio", 0)
        quality_score = metrics.get("instrumental_quality_score", 0)

        table.add_row("Instrumental Coverage", f"{coverage:.1%}",
                     self._assess_coverage(coverage))
        table.add_row("Quality Score", f"{quality_score:.1f}/100",
                     self._get_quality_grade_from_score(quality_score))
        table.add_row("Instrumental Stems", str(metrics.get("num_instrumental_stems", 0)), "-")

        # Show energy distribution
        for key, value in metrics.items():
            if key.endswith("_instrumental_ratio") and not key.startswith("instrumental"):
                stem_name = key.replace("_instrumental_ratio", "")
                table.add_row(f"{stem_name.title()} Ratio", f"{value:.1%}", "-")

        console.print(table)

    def _interactive_checkpoint(self, stage: str, data: Dict):
        """Interactive learning checkpoint."""
        console.print(Panel.fit(
            f"[bold yellow]Checkpoint: {stage}[/bold yellow]\n"
            "Press Enter to continue, 'q' to quit",
            title="🎓 Interactive Mode"
        ))

        choice = input().strip().lower()
        if choice == 'q':
            console.print("[red]Exiting...[/red]")
            exit(0)

    def _save_results(self, results: Dict):
        """Save processing results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.config["output_dir"] / f"instrumental_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def _display_summary(self, results: Dict):
        """Display processing summary."""
        table = Table(title="Processing Summary - Instrumental Separation")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Time", style="yellow")

        for stage, data in results["stages"].items():
            time_taken = data.get("processing_time", "-")
            if isinstance(time_taken, float):
                time_taken = f"{time_taken:.2f}s"
            table.add_row(stage.replace('_', ' ').title(), "✓", str(time_taken))

        table.add_row("Total", "Complete", f"{results['total_time']:.2f}s")
        console.print(table)

        # Show insights
        if results.get("insights"):
            console.print("\n[bold]🎓 Learning Insights:[/bold]")
            for insight in results["insights"]:
                console.print(f"\n[cyan]{insight['stage'].title()}:[/cyan]")
                console.print(insight['content'])

    def _print_stage(self, stage: str, emoji: str):
        """Print stage header."""
        console.print(f"\n{emoji} [bold cyan]{stage}[/bold cyan]")

    # Assessment helpers
    def _assess_energy(self, rms: float) -> str:
        if rms > 0.1:
            return "High"
        elif rms > 0.01:
            return "Medium"
        else:
            return "Low"

    def _assess_dynamic_range(self, dr: float) -> str:
        if dr > 20:
            return "Excellent"
        elif dr > 15:
            return "Good"
        elif dr > 10:
            return "Fair"
        else:
            return "Poor"

    def _assess_brightness(self, centroid: float) -> str:
        if centroid > 3000:
            return "Bright"
        elif centroid > 1500:
            return "Balanced"
        else:
            return "Dark"

    def _assess_readiness(self, score: float) -> str:
        if score > 0.8:
            return "Excellent"
        elif score > 0.6:
            return "Good"
        elif score > 0.4:
            return "Fair"
        else:
            return "Poor"

    def _assess_coverage(self, coverage: float) -> str:
        if coverage > 0.8:
            return "Excellent"
        elif coverage > 0.6:
            return "Good"
        elif coverage > 0.4:
            return "Fair"
        else:
            return "Limited"

    def _get_quality_grade_from_score(self, score: float) -> str:
        if score >= 80:
            return "A+"
        elif score >= 70:
            return "A"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C"
        else:
            return "D"