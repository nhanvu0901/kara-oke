import os
import sys
import subprocess
import platform
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PipelineSetup:
    """Automated setup for the audio processing pipeline."""

    def __init__(self):
        self.root_dir = Path.cwd()
        self.system = platform.system()
        self.machine = platform.machine()
        self.python_version = sys.version_info

    def check_system_requirements(self) -> bool:
        """Check if system meets requirements."""
        logger.info("🔍 Checking system requirements...")

        # Check Python version
        if self.python_version < (3, 9):
            logger.error(f"Python 3.9+ required. Found: {self.python_version.major}.{self.python_version.minor}")
            return False

        # Check for Apple Silicon
        if self.system == "Darwin":
            if "arm" in self.machine.lower():
                logger.info("✅ Apple Silicon detected (M1/M2/M3/M4)")
            else:
                logger.info("✅ Intel Mac detected")
        else:
            logger.warning("⚠️ Non-macOS system detected. Some features may not be optimized.")

        # Check available memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            gb_available = mem.available / (1024 ** 3)
            if gb_available < 8:
                logger.warning(f"⚠️ Low memory: {gb_available:.1f}GB available. 16GB recommended.")
            else:
                logger.info(f"✅ Memory: {gb_available:.1f}GB available")
        except ImportError:
            logger.info("Install psutil for memory checking: pip install psutil")

        return True

    def create_directories(self):
        """Create necessary directory structure."""
        logger.info("📁 Creating directory structure...")

        directories = [
            "models",
            "output",
            "output/separated",
            "output/processed",
            "output/visualizations",
            "output/reports",
            "temp",
            "examples",
            "notebooks",
            "configs"
        ]

        for dir_name in directories:
            dir_path = self.root_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✅ {dir_name}/")

    def install_dependencies(self, dev: bool = False):
        """Install Python dependencies."""
        logger.info("📦 Installing Python dependencies...")

        # Core dependencies
        core_packages = [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "matplotlib>=3.7.0",
            "rich>=13.3.0",
            "requests>=2.28.0",
            "pyyaml>=6.0",
            "python-dotenv>=1.0.0"
        ]

        # Model-specific packages
        model_packages = [
            "demucs>=4.0.0",
            "audiocraft>=1.0.0"
        ]

        # Development packages
        dev_packages = [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "pylint>=2.17.0"
        ]

        all_packages = core_packages + model_packages
        if dev:
            all_packages.extend(dev_packages)

        # Install packages
        for package in all_packages:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    check=True,
                    capture_output=True
                )
                logger.info(f"  ✅ {package.split('>=')[0]}")
            except subprocess.CalledProcessError as e:
                logger.error(f"  ❌ Failed to install {package}: {e}")

    def download_models(self):
        """Pre-download AI models."""
        logger.info("🤖 Downloading AI models...")

        # Download Demucs models
        logger.info("  Downloading Demucs v4 models...")
        try:
            from demucs import pretrained

            models = ["htdemucs", "htdemucs_ft", "htdemucs_6s"]
            for model_name in models:
                logger.info(f"    Downloading {model_name}...")
                pretrained.get_model(model_name)
                logger.info(f"    ✅ {model_name}")
        except Exception as e:
            logger.error(f"  ❌ Failed to download Demucs models: {e}")

        # Download AudioCraft models
        logger.info("  Downloading AudioCraft models...")
        try:
            from audiocraft.models import MusicGen

            models = ["musicgen-small", "musicgen-medium"]
            for model_name in models:
                logger.info(f"    Downloading {model_name}...")
                MusicGen.get_pretrained(model_name)
                logger.info(f"    ✅ {model_name}")
        except Exception as e:
            logger.error(f"  ❌ Failed to download AudioCraft models: {e}")

    def create_config_files(self):
        """Create configuration files."""
        logger.info("⚙️ Creating configuration files...")

        # Default configuration
        default_config = {
            "pipeline": {
                "device": "auto",  # auto, cpu, cuda, mps
                "sample_rate": 44100,
                "output_format": "wav",
                "save_intermediates": True,
                "verbose": True
            },
            "models": {
                "demucs": {
                    "default_model": "htdemucs_ft",
                    "output_format": "wav",
                    "mp3_bitrate": 320,
                    "clip_mode": "rescale",
                    "shifts": 1
                },
                "audiocraft": {
                    "default_model": "musicgen-medium",
                    "temperature": 1.0,
                    "top_k": 250,
                    "top_p": 0.0,
                    "duration": 8.0,
                    "use_sampling": True
                },
                "ddsp": {
                    "checkpoint": "violin",
                    "pitch_shift": 0,
                    "loudness_shift": 0,
                    "f0_confidence_threshold": 0.5
                }
            },
            "analysis": {
                "compute_features": True,
                "feature_types": ["spectral", "rhythm", "harmonic"],
                "visualization": True,
                "export_format": "json"
            },
            "educational": {
                "generate_reports": True,
                "include_explanations": True,
                "difficulty_level": "intermediate",
                "save_checkpoints": True
            },
            "api": {
                "deepseek": {
                    "enabled": False,
                    "model": "deepseek-chat",
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            }
        }

        # Save default config
        config_path = self.root_dir / "configs" / "default_config.json"
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        logger.info(f"  ✅ Created default_config.json")

        # Create .env template
        env_template = """# Environment Variables for Audio Processing Pipeline

# API Keys (optional)
DEEPSEEK_API_KEY=your-api-key-here

# Model Paths (optional, will auto-download if not set)
DEMUCS_MODEL_PATH=
AUDIOCRAFT_MODEL_PATH=

# Processing Settings
DEFAULT_DEVICE=auto
DEFAULT_SAMPLE_RATE=44100
MAX_PROCESSING_LENGTH=600  # Maximum audio length in seconds

# Output Settings
OUTPUT_FORMAT=wav
OUTPUT_QUALITY=high
SAVE_VISUALIZATIONS=true

# Educational Features
EDUCATIONAL_MODE=true
GENERATE_REPORTS=true
INTERACTIVE_MODE=false

# Performance
NUM_WORKERS=4
BATCH_SIZE=1
USE_CACHE=true
"""

        env_path = self.root_dir / ".env.template"
        env_path.write_text(env_template)
        logger.info(f"  ✅ Created .env.template")

        # Create example presets
        presets = {
            "high_quality": {
                "description": "Maximum quality processing",
                "demucs_model": "htdemucs_ft",
                "audiocraft_model": "musicgen-large",
                "shifts": 5,
                "overlap": 0.5
            },
            "fast": {
                "description": "Fast processing for demos",
                "demucs_model": "htdemucs",
                "audiocraft_model": "musicgen-small",
                "shifts": 1,
                "overlap": 0.25
            },
            "educational": {
                "description": "Optimal for learning",
                "demucs_model": "htdemucs_ft",
                "audiocraft_model": "musicgen-medium",
                "save_all_intermediates": True,
                "generate_visualizations": True,
                "verbose": True
            }
        }

        presets_path = self.root_dir / "configs" / "presets.json"
        with open(presets_path, 'w') as f:
            json.dump(presets, f, indent=2)
        logger.info(f"  ✅ Created presets.json")

    def download_examples(self):
        """Download example audio files."""
        logger.info("🎵 Setting up example files...")

        # Create example info file
        examples_info = {
            "examples": [
                {
                    "name": "simple_mix.wav",
                    "description": "Simple 4-track mix for testing",
                    "duration": 30,
                    "genre": "pop",
                    "url": "https://example.com/simple_mix.wav"
                },
                {
                    "name": "complex_orchestra.wav",
                    "description": "Complex orchestral piece",
                    "duration": 60,
                    "genre": "classical",
                    "url": "https://example.com/complex_orchestra.wav"
                }
            ],
            "note": "Replace URLs with actual example files or use your own audio"
        }

        info_path = self.root_dir / "examples" / "examples_info.json"
        with open(info_path, 'w') as f:
            json.dump(examples_info, f, indent=2)
        logger.info(f"  ✅ Created examples_info.json")

        # Create a simple test tone for immediate testing
        try:
            import numpy as np
            import scipy.io.wavfile as wavfile

            # Generate a simple test tone
            sample_rate = 44100
            duration = 5  # seconds
            frequency = 440  # A4 note

            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * frequency * t) * 0.5

            # Add some harmonics
            audio += np.sin(2 * np.pi * frequency * 2 * t) * 0.2
            audio += np.sin(2 * np.pi * frequency * 3 * t) * 0.1

            # Save test file
            test_path = self.root_dir / "examples" / "test_tone.wav"
            wavfile.write(test_path, sample_rate, (audio * 32767).astype(np.int16))
            logger.info(f"  ✅ Created test_tone.wav")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not create test tone: {e}")

    def test_installation(self):
        """Test that everything is working."""
        logger.info("🧪 Testing installation...")

        # Test imports
        test_results = []

        # Test PyTorch
        try:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            test_results.append(("PyTorch", True, f"Device: {device}"))
        except Exception as e:
            test_results.append(("PyTorch", False, str(e)))

        # Test Demucs
        try:
            from demucs import pretrained
            test_results.append(("Demucs", True, "Ready"))
        except Exception as e:
            test_results.append(("Demucs", False, str(e)))

        # Test AudioCraft
        try:
            from audiocraft.models import MusicGen
            test_results.append(("AudioCraft", True, "Ready"))
        except Exception as e:
            test_results.append(("AudioCraft", False, str(e)))

        # Test audio libraries
        try:
            import librosa
            test_results.append(("Librosa", True, f"Version {librosa.__version__}"))
        except Exception as e:
            test_results.append(("Librosa", False, str(e)))

        # Display results
        logger.info("\n📊 Test Results:")
        for component, success, message in test_results:
            if success:
                logger.info(f"  ✅ {component}: {message}")
            else:
                logger.error(f"  ❌ {component}: {message}")

        return all(result[1] for result in test_results)

    def create_shortcuts(self):
        """Create convenient command shortcuts."""
        logger.info("🚀 Creating shortcuts...")

        # Create run script for Unix systems
        if self.system in ["Darwin", "Linux"]:
            run_script = """#!/bin/bash
# Quick run script for audio pipeline

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run with arguments
python main.py "$@"
"""

            run_path = self.root_dir / "run.sh"
            run_path.write_text(run_script)
            run_path.chmod(0o755)
            logger.info(f"  ✅ Created run.sh")

        # Create batch file for Windows
        if self.system == "Windows":
            bat_script = """@echo off
REM Quick run script for audio pipeline

REM Activate virtual environment if it exists
if exist venv\\Scripts\\activate.bat (
    call venv\\Scripts\\activate.bat
)

REM Run with arguments
python main.py %*
"""

            bat_path = self.root_dir / "run.bat"
            bat_path.write_text(bat_script)
            logger.info(f"  ✅ Created run.bat")

    def setup_jupyter(self):
        """Configure Jupyter notebooks."""
        logger.info("📓 Setting up Jupyter notebooks...")

        try:
            # Install Jupyter kernel
            subprocess.run(
                [sys.executable, "-m", "ipykernel", "install", "--user",
                 "--name", "audio_pipeline", "--display-name", "Audio Pipeline"],
                check=True,
                capture_output=True
            )
            logger.info("  ✅ Jupyter kernel installed")

            # Create notebook index
            index_notebook = {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            "# Educational Audio Processing Pipeline - Notebook Index\n",
                            "\n",
                            "## Available Tutorials:\n",
                            "1. [Introduction to Audio Processing](01_introduction.ipynb)\n",
                            "2. [Source Separation with Demucs](02_source_separation.ipynb)\n",
                            "3. [Audio Generation with AudioCraft](03_audiocraft.ipynb)\n",
                            "4. [Style Transfer with DDSP](04_style_transfer.ipynb)\n",
                            "5. [Building Custom Pipelines](05_custom_pipelines.ipynb)\n"
                        ]
                    }
                ],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Audio Pipeline",
                        "language": "python",
                        "name": "audio_pipeline"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 5
            }

            index_path = self.root_dir / "notebooks" / "00_index.ipynb"
            with open(index_path, 'w') as f:
                json.dump(index_notebook, f, indent=2)
            logger.info("  ✅ Created notebook index")

        except Exception as e:
            logger.warning(f"  ⚠️ Jupyter setup incomplete: {e}")

    def display_next_steps(self):
        """Display next steps for the user."""
        print("\n" + "=" * 60)
        print("🎉 Setup Complete!")
        print("=" * 60)
        print("\n📚 Next Steps:")
        print("1. Test the installation:")
        print("   python main.py --input examples/test_tone.wav --mode learning")
        print("\n2. Start Jupyter for interactive learning:")
        print("   jupyter notebook notebooks/00_index.ipynb")
        print("\n3. Process your own audio:")
        print("   python main.py --input your_song.mp3 --mode full")
        print("\n4. Read the documentation:")
        print("   open README.md")
        print("\n5. Configure API keys (optional):")
        print("   cp .env.template .env")
        print("   # Edit .env with your API keys")
        print("\n" + "=" * 60)
        print("Happy Learning! 🎵🤖")
        print("=" * 60 + "\n")

    def run(self, quick: bool = False, dev: bool = False):
        """Run the complete setup process."""
        logger.info("🚀 Starting Audio Pipeline Setup")
        logger.info("=" * 50)

        # Check system
        if not self.check_system_requirements():
            logger.error("System requirements not met!")
            return False

        # Create directories
        self.create_directories()

        # Install dependencies
        if not quick:
            self.install_dependencies(dev=dev)

        # Create config files
        self.create_config_files()

        # Download models (optional, can be slow)
        if not quick:
            response = input("\n📥 Download AI models now? (y/n, can be done later): ")
            if response.lower() == 'y':
                self.download_models()

        # Setup examples
        self.download_examples()

        # Create shortcuts
        self.create_shortcuts()

        # Setup Jupyter
        if dev:
            self.setup_jupyter()

        # Test installation
        success = self.test_installation()

        # Display next steps
        self.display_next_steps()

        return success


def main():
    """Main setup entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Setup Audio Processing Pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Quick setup without downloading models")
    parser.add_argument("--dev", action="store_true",
                        help="Include development tools")
    parser.add_argument("--test-only", action="store_true",
                        help="Only test existing installation")

    args = parser.parse_args()

    setup = PipelineSetup()

    if args.test_only:
        setup.test_installation()
    else:
        success = setup.run(quick=args.quick, dev=args.dev)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()