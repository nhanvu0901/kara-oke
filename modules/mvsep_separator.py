import requests
import time
import os
import torch
import torchaudio
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
import logging
import tempfile
import shutil
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class MVSEPSeparator:
    """
    MVSEP API-based source separator with LLM-guided optimization.
    Supports multiple models and up to 7-stem separation.
    """

    def __init__(self, config: Dict):
        """Initialize MVSEP separator with configuration."""
        self.config = config
        self.base_url = "https://mvsep.com/api/v1"
        self.api_key = os.environ.get("MVSEP_API_KEY")
        if not self.api_key:
            raise ValueError("MVSEP_API_KEY not found in environment variables")
        # Available models with their characteristics
        self.models = {
            # Ensemble models (highest quality)
            "ensemble": {
                "name": "htdemucs_ft",
                "stems": 4,
                "description": "Best overall quality, fine-tuned model"
            },
            "ensemble_extra": {
                "name": "htdemucs_6s",
                "stems": 6,
                "description": "6-stem separation: vocals, drums, bass, piano, guitar, other"
            },

            # Fast models
            "fast": {
                "name": "htdemucs",
                "stems": 4,
                "description": "Faster processing, good quality"
            },
            "ultra_fast": {
                "name": "mdx_extra_q",
                "stems": 4,
                "description": "Fastest processing, decent quality"
            },

            # Specialized models
            "vocal_instrumental": {
                "name": "mdx23c",
                "stems": 2,
                "description": "Optimized for vocal/instrumental separation"
            },
            "karaoke": {
                "name": "uvr_mdx_net_voc_ft",
                "stems": 2,
                "description": "Best for karaoke (removing vocals)"
            },
            "drums_focus": {
                "name": "demucs3_mdx_extra",
                "stems": 4,
                "description": "Enhanced drum separation"
            },

            # Instrument-specific
            "piano": {
                "name": "piano_separator_v1",
                "stems": 2,
                "description": "Specialized for piano isolation"
            },
            "guitar": {
                "name": "guitar_separator_v1",
                "stems": 2,
                "description": "Specialized for guitar isolation"
            },
            "strings": {
                "name": "strings_separator_v1",
                "stems": 2,
                "description": "Specialized for string instruments"
            }
        }

        # API endpoints
        self.endpoints = {
            "upload": f"{self.base_url}/upload",
            "separate": f"{self.base_url}/separate",
            "status": f"{self.base_url}/status",
            "download": f"{self.base_url}/download",
            "models": f"{self.base_url}/models"
        }

        # Processing parameters
        self.timeout = config.get("mvsep", {}).get("api_timeout", 300)
        self.quality = config.get("mvsep", {}).get("quality", "high")
        self.use_llm = config.get("mvsep", {}).get("use_llm_optimization", True)

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Educational-Audio-Pipeline/1.0",
            "Authorization": f"Bearer {self.api_key}"  # Add API key to headers
        })

    def separate(self,
                 audio_path: Path,
                 model: Optional[str] = None,
                 stems: Optional[int] = None,
                 instrument_hints: Optional[Dict] = None,
                 progress_callback: Optional[Callable] = None) -> Dict:
        """
        Separate audio using MVSEP API with intelligent model selection.

        Args:
            audio_path: Path to input audio file
            model: Model to use (auto-selected if None)
            stems: Number of stems (auto-determined if None)
            instrument_hints: LLM classification results for optimization
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with separated stems and metrics
        """
        start_time = time.time()

        try:
            # Step 1: Select optimal model based on LLM hints
            if not model and instrument_hints and self.use_llm:
                model, stems = self._select_optimal_model(instrument_hints)
                logger.info(f"LLM-guided selection: {model} with {stems} stems")
            else:
                model = model or self.config.get("mvsep", {}).get("default_model", "ensemble")
                stems = stems or self.models[model]["stems"]

            if progress_callback:
                progress_callback(0.1, "Uploading audio to MVSEP...")

            # Step 2: Upload audio file
            upload_response = self._upload_audio(audio_path)
            if not upload_response.get("success"):
                raise RuntimeError(f"Upload failed: {upload_response.get('error', 'Unknown error')}")

            file_id = upload_response["file_id"]
            logger.info(f"Audio uploaded successfully. File ID: {file_id}")

            if progress_callback:
                progress_callback(0.2, "Starting separation process...")

            # Step 3: Start separation job
            separation_params = self._prepare_separation_params(
                model, stems, instrument_hints
            )

            job_response = self._start_separation(file_id, separation_params)
            if not job_response.get("success"):
                raise RuntimeError(f"Separation failed to start: {job_response.get('error')}")

            job_id = job_response["job_id"]
            logger.info(f"Separation job started. Job ID: {job_id}")

            # Step 4: Monitor job progress
            result = self._wait_for_completion(job_id, progress_callback)

            if not result.get("success"):
                raise RuntimeError(f"Separation failed: {result.get('error')}")

            if progress_callback:
                progress_callback(0.8, "Downloading separated stems...")

            # Step 5: Download separated stems
            stems_data = self._download_stems(result["stems_urls"])

            # Step 6: Convert to torch tensors
            processed_stems = self._process_stems(stems_data, audio_path)

            # Step 7: Calculate metrics
            metrics = self._calculate_metrics(processed_stems, instrument_hints)

            processing_time = time.time() - start_time

            if progress_callback:
                progress_callback(1.0, "Separation complete!")

            return {
                "stems": processed_stems,
                "source_names": list(processed_stems.keys()),
                "metrics": metrics,
                "processing_time": processing_time,
                "model": model,
                "model_info": self.models[model],
                "job_id": job_id,
                "llm_guided": bool(instrument_hints and self.use_llm)
            }

        except Exception as e:
            logger.error(f"MVSEP separation failed: {e}")
            raise

    def _select_optimal_model(self, instrument_hints: Dict) -> Tuple[str, int]:
        """
        Select optimal model based on LLM instrument classification.

        Args:
            instrument_hints: LLM classification with instruments and confidence

        Returns:
            Tuple of (model_name, num_stems)
        """
        instruments = instrument_hints.get("instruments", [])

        if not instruments:
            return "ensemble", 4

        # Analyze instrument distribution
        has_piano = any(i["name"].lower() in ["piano", "keyboard"] for i in instruments)
        has_guitar = any(i["name"].lower() in ["guitar", "electric guitar", "acoustic guitar"] for i in instruments)
        has_strings = any(i["name"].lower() in ["violin", "cello", "strings", "orchestra"] for i in instruments)
        has_drums = any(i["name"].lower() in ["drums", "percussion"] for i in instruments)
        has_vocals = any(i["name"].lower() in ["vocals", "voice", "singing"] for i in instruments)

        instrument_count = len(instruments)

        # Decision tree for model selection
        if instrument_count <= 2:
            if has_vocals and not has_drums:
                return "vocal_instrumental", 2
            elif has_piano and not has_drums:
                return "piano", 2
            elif has_guitar and not has_drums:
                return "guitar", 2
            else:
                return "fast", 4

        elif instrument_count <= 4:
            if has_drums:
                return "drums_focus", 4
            else:
                return "ensemble", 4

        elif instrument_count <= 6:
            # Use 6-stem model for complex arrangements
            if has_piano and has_guitar:
                return "ensemble_extra", 6
            else:
                return "ensemble", 4

        else:
            # Very complex mix - use best available
            return "ensemble_extra", 6

    def _prepare_separation_params(self,
                                   model: str,
                                   stems: int,
                                   instrument_hints: Optional[Dict]) -> Dict:
        """
        Prepare separation parameters based on model and hints.

        Returns:
            Dictionary of separation parameters
        """
        params = {
            "model": self.models[model]["name"],
            "stems": stems,
            "quality": self.quality,
            "format": "wav",
            "sample_rate": 44100
        }

        # Add frequency hints if available
        if instrument_hints and "instruments" in instrument_hints:
            freq_ranges = []
            for inst in instrument_hints["instruments"]:
                if "freq_range" in inst:
                    freq_ranges.append({
                        "name": inst["name"],
                        "low": inst["freq_range"][0],
                        "high": inst["freq_range"][1]
                    })

            if freq_ranges:
                params["frequency_hints"] = freq_ranges

        # Quality-specific settings
        if self.quality == "high":
            params["shifts"] = 5  # More shifts for better quality
            params["overlap"] = 0.5
        elif self.quality == "medium":
            params["shifts"] = 2
            params["overlap"] = 0.25
        else:  # fast
            params["shifts"] = 1
            params["overlap"] = 0.1

        return params

    def _upload_audio(self, audio_path: Path) -> Dict:
        """Upload audio file to MVSEP API."""
        # This is a mock implementation - replace with actual API
        # In production, you would implement actual file upload
        logger.info(f"Uploading {audio_path.name}...")

        # Mock response
        return {
            "success": True,
            "file_id": f"file_{int(time.time())}",
            "size": audio_path.stat().st_size
        }

    def _start_separation(self, file_id: str, params: Dict) -> Dict:
        """Start separation job on MVSEP."""
        # Mock implementation
        logger.info(f"Starting separation with params: {params}")

        return {
            "success": True,
            "job_id": f"job_{int(time.time())}",
            "estimated_time": 60
        }

    def _wait_for_completion(self,
                             job_id: str,
                             progress_callback: Optional[Callable]) -> Dict:
        """Wait for separation job to complete."""
        # Mock implementation with simulated progress
        for progress in range(30, 80, 10):
            if progress_callback:
                progress_callback(progress / 100, f"Processing... {progress}%")
            time.sleep(0.5)  # Simulate processing time

        # Mock successful completion
        return {
            "success": True,
            "status": "completed",
            "stems_urls": {
                "vocals": f"https://mvsep.com/results/{job_id}/vocals.wav",
                "drums": f"https://mvsep.com/results/{job_id}/drums.wav",
                "bass": f"https://mvsep.com/results/{job_id}/bass.wav",
                "other": f"https://mvsep.com/results/{job_id}/other.wav"
            }
        }

    def _download_stems(self, stems_urls: Dict) -> Dict:
        """Download separated stems from MVSEP."""
        stems_data = {}

        # In production, actually download the files
        # For now, return mock data
        for stem_name, url in stems_urls.items():
            logger.info(f"Downloading {stem_name} from {url}")
            # Mock: would actually download and store the file
            stems_data[stem_name] = f"temp/{stem_name}.wav"

        return stems_data

    def _process_stems(self, stems_data: Dict, original_path: Path) -> Dict:
        """Convert downloaded stems to torch tensors."""
        processed_stems = {}

        # Load original audio for fallback
        original_waveform, sr = torchaudio.load(original_path)

        # In production, load actual downloaded stems
        # For mock, create modified versions of original
        for stem_name in stems_data.keys():
            # Mock: apply basic filtering to simulate stems
            if stem_name == "vocals":
                # Simulate vocal extraction with high-pass filter
                processed_stems[stem_name] = original_waveform * 0.3
            elif stem_name == "drums":
                # Simulate drum extraction
                processed_stems[stem_name] = original_waveform * 0.2
            elif stem_name == "bass":
                # Simulate bass extraction with low-pass
                processed_stems[stem_name] = original_waveform * 0.25
            else:
                # Other instruments
                processed_stems[stem_name] = original_waveform * 0.25

        return processed_stems

    def _calculate_metrics(self,
                           stems: Dict,
                           instrument_hints: Optional[Dict]) -> Dict:
        """Calculate separation quality metrics."""
        metrics = {
            "num_stems": len(stems),
            "model_confidence": 0.85  # Mock confidence
        }

        # Energy distribution
        total_energy = 0
        for stem_name, stem_audio in stems.items():
            energy = torch.mean(stem_audio ** 2).item()
            metrics[f"{stem_name}_energy"] = energy
            total_energy += energy

        # Normalize energies
        for stem_name in stems.keys():
            metrics[f"{stem_name}_energy_ratio"] = (
                    metrics[f"{stem_name}_energy"] / (total_energy + 1e-10)
            )

        # Add LLM confidence if available
        if instrument_hints:
            avg_confidence = sum(
                inst.get("confidence", 0)
                for inst in instrument_hints.get("instruments", [])
            ) / max(len(instrument_hints.get("instruments", [1])), 1)
            metrics["llm_confidence"] = avg_confidence

        return metrics

    def get_available_models(self) -> List[Dict]:
        """Get list of available models and their capabilities."""
        return [
            {
                "id": key,
                "name": value["name"],
                "stems": value["stems"],
                "description": value["description"]
            }
            for key, value in self.models.items()
        ]

    def cleanup(self):
        """Clean up resources and temporary files."""
        self.session.close()
        # Clean temporary files if any
        temp_dir = Path("temp")
        if temp_dir.exists():
            for file in temp_dir.glob("*.wav"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {file}: {e}")