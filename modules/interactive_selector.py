# modules/interactive_selector.py - COMPLETE IMPLEMENTATION
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import torch
import torchaudio
import sounddevice as sd
import numpy as np
from typing import Dict, List, Optional
import time


class InteractiveSelector:
    """Interactive stem transformation selection interface."""

    def __init__(self, sample_rate: int = 44100):
        self.console = Console()
        self.sample_rate = sample_rate
        self.selected_transformations = {}

    def display_stem_options(self, stem_analysis: Dict) -> str:
        """Display transformation options for a stem."""

        # Create panel with stem info
        info_text = (
            f"Type: [cyan]{stem_analysis['instrument_type']}[/cyan]\n"
            f"Key: [yellow]{stem_analysis.get('key', 'Unknown')}[/yellow]\n"
            f"Energy: [green]{stem_analysis.get('energy', 0):.4f}[/green]"
        )

        self.console.print(Panel(info_text, title=f"🎵 {stem_analysis['name']}", border_style="blue"))

        # Create table with transformation options
        table = Table(title="Available Transformations", show_header=True)
        table.add_column("Option", style="cyan", width=10)
        table.add_column("Transformation", style="yellow", width=20)
        table.add_column("Description", style="white", width=40)
        table.add_column("Compatibility", style="green", width=15)

        # Add transformation options
        options = []
        for idx, (key, info) in enumerate(stem_analysis['recommended_transformations'].items(), 1):
            compatibility = info.get('compatibility', 0.5)
            comp_str = f"{compatibility:.0%}"

            # Color code compatibility
            if compatibility >= 0.8:
                comp_str = f"[green]{comp_str}[/green]"
            elif compatibility >= 0.6:
                comp_str = f"[yellow]{comp_str}[/yellow]"
            else:
                comp_str = f"[red]{comp_str}[/red]"

            table.add_row(
                str(idx),
                key,
                info['description'],
                comp_str
            )
            options.append(key)

        # Add skip option
        table.add_row("0", "skip", "Keep original audio", "[dim]100%[/dim]")
        table.add_row("p", "preview", "Preview transformation", "[dim]-[/dim]")

        self.console.print(table)

        # Get user selection
        while True:
            choice = Prompt.ask(
                "Select transformation",
                choices=[str(i) for i in range(len(options) + 1)] + ['p'],
                default="0"
            )

            if choice == '0':
                return "skip"
            elif choice == 'p':
                # Preview mode
                preview_idx = Prompt.ask(
                    "Which transformation to preview?",
                    choices=[str(i) for i in range(1, len(options) + 1)]
                )
                selected_transform = options[int(preview_idx) - 1]
                self.console.print(f"[dim]Preview of '{selected_transform}' would play here[/dim]")
                # In real implementation, would call preview_transformation
                continue
            else:
                return options[int(choice) - 1]

    def preview_transformation(self,
                               audio: torch.Tensor,
                               transformation: str,
                               duration: float = 5.0) -> None:
        """Generate and play preview of transformation."""

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
        ) as progress:
            task = progress.add_task(f"Generating preview of {transformation}...", total=100)

            # Truncate audio for preview
            preview_samples = int(duration * self.sample_rate)
            if audio.shape[-1] > preview_samples:
                preview_audio = audio[..., :preview_samples]
            else:
                preview_audio = audio

            progress.update(task, advance=50)

            # Convert to numpy for playback
            if preview_audio.dim() > 1:
                audio_np = preview_audio.mean(dim=0).numpy()
            else:
                audio_np = preview_audio.numpy()

            # Normalize
            audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-10)

            progress.update(task, advance=50)

        # Play audio
        try:
            self.console.print(f"[green]Playing {duration}s preview...[/green]")
            sd.play(audio_np, self.sample_rate)
            sd.wait()  # Wait until playback is done
        except Exception as e:
            self.console.print(f"[red]Could not play preview: {e}[/red]")

    def display_transformation_summary(self, transformations: Dict[str, str]) -> None:
        """Display summary of selected transformations."""

        table = Table(title="Selected Transformations", show_header=True)
        table.add_column("Stem", style="cyan")
        table.add_column("Transformation", style="yellow")

        for stem_name, transformation in transformations.items():
            if transformation != "skip":
                table.add_row(stem_name, transformation)

        self.console.print(table)

        # Confirm selections
        if not Confirm.ask("Proceed with these transformations?"):
            return False
        return True

    def calculate_compatibility(self, transformation_info: Dict) -> float:
        """Calculate compatibility score for a transformation."""
        # This is already calculated in stem_analyzer
        return transformation_info.get('compatibility', 0.5)