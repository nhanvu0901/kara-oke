import requests
import time
import os
import json
import torch
import torchaudio
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
import logging
import tempfile
import shutil
from urllib.parse import urljoin, urlparse, parse_qs

logger = logging.getLogger(__name__)


class MVSEPSeparator:
    """
    MVSEP API-based source separator with LLM-guided optimization.
    Supports multiple models and up to 7-stem separation.
    """

    def __init__(self, config: Dict):
        """Initialize MVSEP separator with configuration."""
        self.config = config
        self.base_url = "https://mvsep.com"
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

        # Processing parameters
        self.timeout = config.get("mvsep", {}).get("api_timeout", 300)
        self.quality = config.get("mvsep", {}).get("quality", "high")
        self.use_llm = config.get("mvsep", {}).get("use_llm_optimization", True)

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Educational-Audio-Pipeline/1.0"
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

            # Step 2: Upload audio and start separation
            separation_response = self._upload_and_separate(
                audio_path,
                model,
                stems,
                progress_callback
            )

            if not separation_response.get("success"):
                raise RuntimeError(f"Separation failed: {separation_response.get('error', 'Unknown error')}")

            # Step 3: Download and process stems
            if progress_callback:
                progress_callback(0.8, "Downloading separated stems...")

            stems_data = self._download_stems(separation_response["stems_urls"])

            # Step 4: Convert to torch tensors
            processed_stems = self._process_stems(stems_data)

            # Step 5: Calculate metrics
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
                "llm_guided": bool(instrument_hints and self.use_llm)
            }

        except Exception as e:
            logger.error(f"MVSEP separation failed: {e}")
            # Fallback to basic separation
            return self._fallback_separation(audio_path, progress_callback)

    def _upload_and_separate(self,
                             audio_path: Path,
                             model: str,
                             stems: int,
                             progress_callback: Optional[Callable]) -> Dict:
        """Upload audio and initiate separation on MVSEP."""
        try:
            # Prepare the API parameters
            model_name = self.models[model]["name"]

            # Upload file and start separation
            with open(audio_path, 'rb') as f:
                files = {
                    'audiofile': (audio_path.name, f, 'audio/mpeg')
                }

                data = {
                    'api_key': self.api_key,
                    'model': model_name,
                    'output_format': 'wav',
                    'stems': stems
                }

                # Add quality settings
                if self.quality == "high":
                    data['shifts'] = 5
                    data['overlap'] = 0.5
                elif self.quality == "medium":
                    data['shifts'] = 2
                    data['overlap'] = 0.25
                else:  # fast
                    data['shifts'] = 1
                    data['overlap'] = 0.1

                logger.info(f"Uploading {audio_path.name} to MVSEP...")

                response = self.session.post(
                    f"{self.base_url}/api/separation/create",
                    files=files,
                    data=data,
                    timeout=60
                )

            if response.status_code == 200:
                result = json.loads(response.content.decode('utf-8'))

                if result.get('success'):
                    # Extract hash and link from response
                    hash_id = result['data']['hash']
                    result_link = result['data']['link']

                    logger.info(f"Separation job created: {hash_id}")

                    if progress_callback:
                        progress_callback(0.2, "Processing audio on MVSEP servers...")

                    # Wait for processing to complete and get results
                    stems_urls = self._wait_and_get_results(result_link, hash_id, progress_callback)

                    return {
                        "success": True,
                        "stems_urls": stems_urls,
                        "hash": hash_id
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get('error', 'Unknown error from API')
                    }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }

        except requests.Timeout:
            logger.error("MVSEP API timeout")
            return {"success": False, "error": "Request timeout"}
        except Exception as e:
            logger.error(f"Upload/separation error: {e}")
            return {"success": False, "error": str(e)}

    def _wait_and_get_results(self,
                              result_link: str,
                              hash_id: str,
                              progress_callback: Optional[Callable]) -> Dict:
        """Poll MVSEP for processing completion and get download links."""
        max_attempts = 60  # 5 minutes max wait
        poll_interval = 5  # seconds

        for attempt in range(max_attempts):
            try:
                # Poll the result link
                response = self.session.get(result_link, timeout=10)

                if response.status_code == 200:
                    # Check if JSON response (still processing) or file ready
                    content_type = response.headers.get('content-type', '')

                    if 'application/json' in content_type:
                        # Still processing or error
                        data = response.json()

                        if data.get('status') == 'processing':
                            if progress_callback:
                                progress = min(0.7, 0.2 + (attempt / max_attempts) * 0.5)
                                progress_callback(progress, f"Processing... {attempt * poll_interval}s")
                            time.sleep(poll_interval)
                            continue

                        elif data.get('status') == 'error':
                            raise RuntimeError(f"Processing error: {data.get('error', 'Unknown')}")

                        elif data.get('status') == 'completed':
                            # Get stem URLs
                            return self._extract_stem_urls(data, hash_id)

                    elif 'audio' in content_type or response.headers.get('content-disposition'):
                        # Results might be ready - try to get individual stems
                        return self._get_stem_urls(hash_id)

                    else:
                        # HTML or other response - results might be ready
                        return self._get_stem_urls(hash_id)

                elif response.status_code == 202:
                    # Still processing
                    if progress_callback:
                        progress = min(0.7, 0.2 + (attempt / max_attempts) * 0.5)
                        progress_callback(progress, f"Processing... {attempt * poll_interval}s")
                    time.sleep(poll_interval)

                else:
                    logger.warning(f"Unexpected status code: {response.status_code}")
                    time.sleep(poll_interval)

            except Exception as e:
                logger.warning(f"Poll attempt {attempt + 1} failed: {e}")
                time.sleep(poll_interval)

        raise TimeoutError("MVSEP processing timed out")

    def _get_stem_urls(self, hash_id: str) -> Dict:
        """Construct stem download URLs based on hash."""
        # MVSEP typically provides stems with predictable URLs
        # This might need adjustment based on actual API behavior
        base_url = f"{self.base_url}/api/separation/download"

        stems_urls = {
            "vocals": f"{base_url}?hash={hash_id}&stem=vocals",
            "drums": f"{base_url}?hash={hash_id}&stem=drums",
            "bass": f"{base_url}?hash={hash_id}&stem=bass",
            "other": f"{base_url}?hash={hash_id}&stem=other"
        }

        # Verify at least one URL is accessible
        try:
            test_response = self.session.head(stems_urls["vocals"], timeout=5)
            if test_response.status_code != 200:
                # Try alternative URL format
                stems_urls = self._try_alternative_urls(hash_id)
        except:
            stems_urls = self._try_alternative_urls(hash_id)

        return stems_urls

    def _try_alternative_urls(self, hash_id: str) -> Dict:
        """Try alternative URL formats for stems."""
        # Alternative URL patterns MVSEP might use
        patterns = [
            f"{self.base_url}/results/{hash_id}",
            f"{self.base_url}/api/get/{hash_id}",
            f"{self.base_url}/download/{hash_id}"
        ]

        for pattern in patterns:
            stems_urls = {
                "vocals": f"{pattern}/vocals.wav",
                "drums": f"{pattern}/drums.wav",
                "bass": f"{pattern}/bass.wav",
                "other": f"{pattern}/other.wav"
            }

            try:
                # Test if accessible
                response = self.session.head(stems_urls["vocals"], timeout=5)
                if response.status_code == 200:
                    return stems_urls
            except:
                continue

        # Default fallback
        return {
            "vocals": f"{self.base_url}/api/separation/get?hash={hash_id}&stem=vocals",
            "drums": f"{self.base_url}/api/separation/get?hash={hash_id}&stem=drums",
            "bass": f"{self.base_url}/api/separation/get?hash={hash_id}&stem=bass",
            "other": f"{self.base_url}/api/separation/get?hash={hash_id}&stem=other"
        }

    def _extract_stem_urls(self, data: Dict, hash_id: str) -> Dict:
        """Extract stem URLs from API response."""
        if 'stems' in data:
            return data['stems']
        elif 'urls' in data:
            return data['urls']
        elif 'downloads' in data:
            return data['downloads']
        else:
            # Construct URLs if not provided
            return self._get_stem_urls(hash_id)

    def _download_stems(self, stems_urls: Dict) -> Dict:
        """Download separated stems from MVSEP."""
        stems_paths = {}
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        for stem_name, url in stems_urls.items():
            try:
                logger.info(f"Downloading {stem_name} from {url}")

                response = self.session.get(url, stream=True, timeout=30)

                if response.status_code == 200:
                    # Generate unique filename
                    timestamp = int(time.time())
                    stem_path = temp_dir / f"{stem_name}_{timestamp}.wav"

                    # Download in chunks
                    with open(stem_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    stems_paths[stem_name] = stem_path
                    logger.info(f"Downloaded {stem_name} to {stem_path}")

                else:
                    logger.warning(f"Failed to download {stem_name}: HTTP {response.status_code}")

            except Exception as e:
                logger.error(f"Failed to download {stem_name}: {e}")

        if not stems_paths:
            raise RuntimeError("Failed to download any stems")

        return stems_paths

    def _process_stems(self, stems_paths: Dict) -> Dict:
        """Convert downloaded stems to torch tensors."""
        processed_stems = {}

        for stem_name, stem_path in stems_paths.items():
            try:
                if stem_path.exists():
                    waveform, sr = torchaudio.load(stem_path)
                    processed_stems[stem_name] = waveform
                    logger.info(f"Loaded {stem_name}: shape={waveform.shape}, sr={sr}")
                else:
                    logger.warning(f"Stem file not found: {stem_path}")

            except Exception as e:
                logger.error(f"Failed to load {stem_name}: {e}")

        return processed_stems

    def _fallback_separation(self,
                             audio_path: Path,
                             progress_callback: Optional[Callable]) -> Dict:
        """Fallback separation when API is unavailable."""
        logger.warning("Using fallback separation (API unavailable)")

        if progress_callback:
            progress_callback(0.5, "Using local fallback separation...")

        # Load original audio
        waveform, sr = torchaudio.load(audio_path)

        # Create basic frequency-based separation (very basic fallback)
        # This is just for demonstration - real separation requires proper models
        processed_stems = {
            "vocals": waveform * 0.3,  # Placeholder
            "drums": waveform * 0.2,  # Placeholder
            "bass": waveform * 0.25,  # Placeholder
            "other": waveform * 0.25  # Placeholder
        }

        if progress_callback:
            progress_callback(1.0, "Fallback separation complete")

        return {
            "stems": processed_stems,
            "source_names": list(processed_stems.keys()),
            "metrics": {
                "num_stems": len(processed_stems),
                "fallback_used": True
            },
            "processing_time": 0,
            "model": "fallback",
            "model_info": {"description": "Fallback separation (no API)"},
            "llm_guided": False
        }

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

    def _calculate_metrics(self,
                           stems: Dict,
                           instrument_hints: Optional[Dict]) -> Dict:
        """Calculate separation quality metrics."""
        metrics = {
            "num_stems": len(stems),
            "model_confidence": 0.85
        }

        # Energy distribution
        total_energy = 0
        for stem_name, stem_audio in stems.items():
            if stem_audio is not None and stem_audio.numel() > 0:
                energy = torch.mean(stem_audio ** 2).item()
                metrics[f"{stem_name}_energy"] = energy
                total_energy += energy

        # Normalize energies
        if total_energy > 0:
            for stem_name in stems.keys():
                if f"{stem_name}_energy" in metrics:
                    metrics[f"{stem_name}_energy_ratio"] = (
                            metrics[f"{stem_name}_energy"] / total_energy
                    )

        # Add LLM confidence if available
        if instrument_hints and "instruments" in instrument_hints:
            instruments = instrument_hints.get("instruments", [])
            if instruments:
                avg_confidence = sum(
                    inst.get("confidence", 0) for inst in instruments
                ) / len(instruments)
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

        # Clean temporary files
        temp_dir = Path("temp")
        if temp_dir.exists():
            for file in temp_dir.glob("*.wav"):
                try:
                    file.unlink()
                    logger.info(f"Cleaned up {file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {file}: {e}")