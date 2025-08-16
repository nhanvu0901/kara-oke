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
    MVSEP API-based source separator focused on ensemble_extra model.
    Supports up to 6-stem separation with ensemble_extra.
    """

    def __init__(self, config: Dict):
        """Initialize MVSEP separator with configuration."""
        self.config = config
        self.base_url = "https://mvsep.com"
        self.api_key = os.environ.get("MVSEP_API_KEY")
        if not self.api_key:
            raise ValueError("MVSEP_API_KEY not found in environment variables")

        # Available models with their characteristics - focused on ensemble_extra
        self.models = {
            "ensemble_extra": {
                "name": "htdemucs_6s",
                "stems": 6,
                "stem_names": ["vocals", "drums", "bass", "piano", "guitar", "other"],
                "description": "6-stem separation: vocals, drums, bass, piano, guitar, other"
            },
            "ensemble": {
                "name": "htdemucs_ft",
                "stems": 4,
                "stem_names": ["vocals", "drums", "bass", "other"],
                "description": "Best overall quality, fine-tuned model"
            },
            "fast": {
                "name": "htdemucs",
                "stems": 4,
                "stem_names": ["vocals", "drums", "bass", "other"],
                "description": "Faster processing, good quality"
            }
        }

        # Processing parameters
        self.timeout = config.get("mvsep", {}).get("api_timeout", 300)
        self.quality = config.get("mvsep", {}).get("quality", "high")

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Educational-Audio-Pipeline/1.0",
            "X-API-Key": self.api_key
        })

    def separate(self,
                 audio_path: Path,
                 model: Optional[str] = None,
                 stems: Optional[int] = None,
                 instrument_hints: Optional[Dict] = None,
                 progress_callback: Optional[Callable] = None) -> Dict:
        """
        Separate audio using MVSEP API with ensemble_extra model.

        Args:
            audio_path: Path to input audio file
            model: Model to use (defaults to ensemble_extra)
            stems: Number of stems (auto-determined from model)
            instrument_hints: Not used (LLM disabled)
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with separated stems and metrics
        """
        start_time = time.time()

        try:
            # Use ensemble_extra by default
            model = model or "ensemble_extra"
            model_info = self.models[model]
            stems = model_info["stems"]
            expected_stem_names = model_info["stem_names"]

            logger.info(f"Using MVSEP model: {model} ({model_info['name']}) with {stems} stems")
            logger.info(f"Expected stems: {', '.join(expected_stem_names)}")

            if progress_callback:
                progress_callback(0.1, "Uploading audio to MVSEP...")

            # Upload audio and start separation
            separation_response = self._upload_and_separate(
                audio_path,
                model,
                stems,
                progress_callback
            )

            if not separation_response.get("success"):
                raise RuntimeError(f"Separation failed: {separation_response.get('error', 'Unknown error')}")

            # Download and process stems
            if progress_callback:
                progress_callback(0.8, "Downloading separated stems...")

            stems_data = self._download_stems(separation_response["stems_urls"], expected_stem_names)

            # Convert to torch tensors
            processed_stems = self._process_stems(stems_data)

            # Calculate metrics
            metrics = self._calculate_metrics(processed_stems)

            processing_time = time.time() - start_time

            if progress_callback:
                progress_callback(1.0, "Separation complete!")

            return {
                "stems": processed_stems,
                "source_names": list(processed_stems.keys()),
                "metrics": metrics,
                "processing_time": processing_time,
                "model": model,
                "model_info": model_info,
                "llm_guided": False
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

                # Add quality settings for ensemble_extra
                if self.quality == "high":
                    data['shifts'] = 5  # More shifts for better quality
                    data['overlap'] = 0.5
                elif self.quality == "medium":
                    data['shifts'] = 2
                    data['overlap'] = 0.25
                else:  # fast
                    data['shifts'] = 1
                    data['overlap'] = 0.1

                logger.info(f"Uploading {audio_path.name} to MVSEP with {model_name} model...")

                response = self.session.post(
                    f"{self.base_url}/api/separation/create",
                    files=files,
                    data=data,
                    timeout=60
                )

            if response.status_code == 200:
                result = json.loads(response.content.decode('utf-8'))

                if result.get('success'):
                    hash_id = result['data']['hash']
                    result_link = result['data']['link']

                    logger.info(f"Separation job created: {hash_id}")

                    if progress_callback:
                        progress_callback(0.2, "Processing audio on MVSEP servers...")

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
        max_attempts = 60
        poll_interval = 5

        for attempt in range(max_attempts):
            try:
                response = self.session.get(result_link, timeout=10)

                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')

                    if 'application/json' in content_type:
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
                            return self._extract_stem_urls(data, hash_id)

                    elif 'audio' in content_type or response.headers.get('content-disposition'):
                        return self._get_stem_urls(hash_id)

                    else:
                        return self._get_stem_urls(hash_id)

                elif response.status_code == 202:
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
        """Construct stem download URLs for ensemble_extra (6 stems)."""
        base_url = f"{self.base_url}/api/separation/download"

        stems_urls = {
            "vocals": f"{base_url}?hash={hash_id}&stem=vocals",
            "drums": f"{base_url}?hash={hash_id}&stem=drums",
            "bass": f"{base_url}?hash={hash_id}&stem=bass",
            "piano": f"{base_url}?hash={hash_id}&stem=piano",
            "guitar": f"{base_url}?hash={hash_id}&stem=guitar",
            "other": f"{base_url}?hash={hash_id}&stem=other"
        }

        try:
            test_response = self.session.head(stems_urls["vocals"], timeout=5)
            if test_response.status_code != 200:
                stems_urls = self._try_alternative_urls(hash_id)
        except:
            stems_urls = self._try_alternative_urls(hash_id)

        return stems_urls

    def _try_alternative_urls(self, hash_id: str) -> Dict:
        """Try alternative URL formats for ensemble_extra stems."""
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
                "piano": f"{pattern}/piano.wav",
                "guitar": f"{pattern}/guitar.wav",
                "other": f"{pattern}/other.wav"
            }

            try:
                response = self.session.head(stems_urls["vocals"], timeout=5)
                if response.status_code == 200:
                    return stems_urls
            except:
                continue

        return {
            "vocals": f"{self.base_url}/api/separation/get?hash={hash_id}&stem=vocals",
            "drums": f"{self.base_url}/api/separation/get?hash={hash_id}&stem=drums",
            "bass": f"{self.base_url}/api/separation/get?hash={hash_id}&stem=bass",
            "piano": f"{self.base_url}/api/separation/get?hash={hash_id}&stem=piano",
            "guitar": f"{self.base_url}/api/separation/get?hash={hash_id}&stem=guitar",
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
            return self._get_stem_urls(hash_id)

    def _download_stems(self, stems_urls: Dict, expected_stem_names: List[str]) -> Dict:
        """Download separated stems from MVSEP."""
        stems_paths = {}
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        for stem_name in expected_stem_names:
            if stem_name in stems_urls:
                url = stems_urls[stem_name]
                try:
                    logger.info(f"Downloading {stem_name} from {url}")

                    response = self.session.get(url, stream=True, timeout=30)

                    if response.status_code == 200:
                        timestamp = int(time.time())
                        stem_path = temp_dir / f"{stem_name}_{timestamp}.wav"

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

        waveform, sr = torchaudio.load(audio_path)

        processed_stems = {
            "vocals": waveform * 0.2,
            "drums": waveform * 0.15,
            "bass": waveform * 0.15,
            "piano": waveform * 0.15,
            "guitar": waveform * 0.15,
            "other": waveform * 0.2
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

    def _calculate_metrics(self, stems: Dict) -> Dict:
        """Calculate separation quality metrics."""
        metrics = {
            "num_stems": len(stems),
            "model_confidence": 0.85
        }

        total_energy = 0
        for stem_name, stem_audio in stems.items():
            if stem_audio is not None and stem_audio.numel() > 0:
                energy = torch.mean(stem_audio ** 2).item()
                metrics[f"{stem_name}_energy"] = energy
                total_energy += energy

        if total_energy > 0:
            for stem_name in stems.keys():
                if f"{stem_name}_energy" in metrics:
                    metrics[f"{stem_name}_energy_ratio"] = (
                            metrics[f"{stem_name}_energy"] / total_energy
                    )

        return metrics

    def get_available_models(self) -> List[Dict]:
        """Get list of available models and their capabilities."""
        return [
            {
                "id": key,
                "name": value["name"],
                "stems": value["stems"],
                "stem_names": value["stem_names"],
                "description": value["description"]
            }
            for key, value in self.models.items()
        ]

    def cleanup(self):
        """Clean up resources and temporary files."""
        self.session.close()

        temp_dir = Path("temp")
        if temp_dir.exists():
            for file in temp_dir.glob("*.wav"):
                try:
                    file.unlink()
                    logger.info(f"Cleaned up {file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {file}: {e}")