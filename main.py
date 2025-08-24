#!/usr/bin/env python3
"""
Educational Audio Processing Pipeline - Instrumental Separation
================================================================
Clean, focused entry point for audio instrumental source separation learning.
Processes audio into instrumental stems only (bass, drums, other).
"""

import argparse
import sys
from pathlib import Path
import torch
from rich.console import Console
from rich.panel import Panel

from modules.pipeline import SeparationPipeline
from modules.config import load_config, create_default_config

console = Console()


def main():
    """Main entry point for instrumental separation."""
    parser = argparse.ArgumentParser(
        description="Educational Audio Instrumental Separation Pipeline - Extract bass, drums, and other instrumental stems"
    )

    # Core arguments
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input audio file to separate into instrumental stems")
    parser.add_argument("--output", "-o", type=Path, default=Path("output"),
                        help="Output directory for instrumental stems")
    parser.add_argument("--mode", "-m", choices=["full", "learning", "separator", "analysis"],
                        default="learning",
                        help="Processing mode (learning includes educational insights)")

    # Model settings
    parser.add_argument("--model", default="htdemucs_6s",
                        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s"],
                        help="Demucs model for instrumental separation")
    parser.add_argument("--shifts", type=int, default=10,
                        help="Separation quality - higher = better quality (1-10)")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Overlap factor for processing (0.1-0.5)")

    # Quality presets
    parser.add_argument("--fast", action="store_true",
                        help="Fast processing for quick instrumental extraction")
    parser.add_argument("--high-quality", action="store_true",
                        help="High quality instrumental separation (slower)")

    # Features
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive learning mode with checkpoints")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualizations to speed up processing")
    parser.add_argument("--batch", type=Path,
                        help="Batch process directory of audio files")

    args = parser.parse_args()

    # Validate input
    if not args.input.exists() and not args.batch:
        console.print(f"[red]Error: {args.input} not found[/red]")
        sys.exit(1)

    # Print banner
    console.print(Panel.fit(
        "[bold magenta]Educational Instrumental Audio Separation Pipeline[/bold magenta]\n"
        "[yellow]Extracts: Bass, Drums, and Other instrumental stems[/yellow]\n"
        f"Input: {args.input if not args.batch else args.batch}\n"
        f"Mode: {args.mode}\n"
        f"Model: {args.model}",
        title="🎸 Instrumental Separation Lab 🥁"
    ))

    # Load configuration
    config = create_default_config()
    config.update({
        "output_dir": args.output,
        "visualization": not args.no_viz,
        "demucs": {
            "model": args.model,
            "shifts": 3 if args.fast else 10 if args.high_quality else args.shifts,
            "overlap": args.overlap
        }
    })

    # Initialize and run pipeline
    try:
        console.print("\n[cyan]Initializing instrumental separation pipeline...[/cyan]")
        pipeline = SeparationPipeline(config)

        if args.batch:
            console.print(f"\n[cyan]Batch processing instrumental separation for: {args.batch}[/cyan]")
            pipeline.process_batch(args.batch, args.mode, args.interactive)
        else:
            console.print(f"\n[cyan]Processing instrumental separation for: {args.input}[/cyan]")
            pipeline.process_audio(args.input, args.mode, args.interactive)

        console.print("\n[bold green]✨ Instrumental separation complete![/bold green]")
        console.print("[yellow]Output contains: bass, drums, and other instrumental stems[/yellow]")
        console.print(f"[dim]Check the {args.output}/instrumentals/ directory for your stems[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error during instrumental separation: {e}[/red]")
        console.print("[dim]Tip: Try using --fast mode for quicker processing[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()