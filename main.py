#!/usr/bin/env python3
"""
Educational Audio Processing Pipeline - Simple Main
==================================================
Clean, focused entry point for audio source separation learning.
"""

import argparse
import sys
from pathlib import Path
import torch
from rich.console import Console
from rich.panel import Panel

from modules.pipeline import SeparationPipeline
from modules.config import  create_default_config

console = Console()


def main():
    """Simple main entry point."""
    parser = argparse.ArgumentParser(description="Educational Audio Separation Pipeline")

    # Core arguments
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input audio file")
    parser.add_argument("--output", "-o", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--mode", "-m", choices=["full", "learning", "separator", "analysis"],
                        default="learning", help="Processing mode")

    # Model settings
    parser.add_argument("--model", default="htdemucs_6s",
                        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s"], help="Demucs model")
    parser.add_argument("--shifts", type=int, default=10, help="Separation quality (1-10)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap factor (0.1-0.5)")

    # Quality presets
    parser.add_argument("--fast", action="store_true", help="Fast processing")
    parser.add_argument("--high-quality", action="store_true", help="High quality processing")

    parser.add_argument("--separator", default="uvr5",
                        choices=["uvr5", "demucs", "hybrid"],
                        help="Separation engine")
    # Features
    parser.add_argument("--interactive", action="store_true", help="Interactive learning mode")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualizations")
    parser.add_argument("--batch", type=Path, help="Batch process directory")

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        console.print(f"[red]Error: {args.input} not found[/red]")
        sys.exit(1)

    # Print banner
    console.print(Panel.fit(
        "[bold magenta]Educational Audio Separation Pipeline[/bold magenta]\n"
        f"Input: {args.input}\n"
        f"Mode: {args.mode}\n"
        f"Model: {args.model}",
        title="🎵 Audio Learning Lab"
    ))

    # Load configuration
    config = create_default_config()
    config.update({
        "output_dir": args.output,
        "visualization": not args.no_viz,
        "separator_engine": args.separator,
    })

    # Initialize and run pipeline
    try:
        pipeline = SeparationPipeline(config)

        if args.batch:
            pipeline.process_batch(args.batch, args.mode, args.interactive)
        else:
            pipeline.process_audio(args.input, args.mode, args.interactive)

        console.print("\n[bold green]✨ Processing complete![/bold green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()