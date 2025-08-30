"""
Configuration Management
========================
Handle all pipeline configuration settings.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for the pipeline."""
    return {
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "sample_rate": 44100,
        "output_dir": Path("output"),
        "temp_dir": Path("temp"),
        "educational_mode": True,
        "verbose": True,
        "visualization": True,
        "analysis": True,
        "demucs": {
            "model": "htdemucs_6s",
            "shifts": 15,
            "overlap": 0.75,
        },
        "deepseek": {
            "api_key": os.environ.get("DEEPSEEK_API_KEY"),
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "uvr5": {
            "model": "UVR-MDX-NET-Inst_HQ_3",
            "chunk_size": 524288,
            "overlap": 0.75,
            "margin": 88200,
            "denoise": True,
            "post_process": True,
            "tta": True,
            "separation_mode": "enhanced",  # Add this
            "target_instruments": ["drums", "bass", "piano", "guitar", "strings", "other"]
        }
    }





def save_config(config: Dict[str, Any], config_path: Path):
    """Save configuration to file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings for JSON serialization
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, Path):
            serializable_config[key] = str(value)
        elif isinstance(value, dict):
            serializable_config[key] = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in value.items()
            }
        else:
            serializable_config[key] = value

    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def get_quality_preset(preset_name: str) -> Dict[str, Any]:
    """Get predefined quality presets."""
    presets = {
        "fast": {
            "demucs": {
                "model": "htdemucs_ft",
                "shifts": 3,
                "overlap": 0.25
            }
        },
        "balanced": {
            "demucs": {
                "model": "htdemucs_6s",
                "shifts": 10,
                "overlap": 0.5
            }
        },
        "high_quality": {
            "demucs": {
                "model": "htdemucs_6s",
                "shifts": 10,
                "overlap": 0.5
            }
        }
    }

    return presets.get(preset_name, presets["balanced"])


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries."""
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged