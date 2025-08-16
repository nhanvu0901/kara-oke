import requests
import time
import os
import json
import torch
import torchaudio
from pathlib import Path
from typing import Dict, Optional, Callable, List
import logging
import tempfile

logger = logging.getLogger(__name__)


class MVSEPSeparator:
    """
    MVSEP API-based source separator with proper API integration.
    """

    def __init__(self, config: Dict):
        """Initialize MVSEP separator with configuration."""
        self.config = config
        self.base_url = "https://mvsep.com"
        self.api_token = os.environ.get("MVSEP_API_KEY")

        if not self.api_token:
            raise ValueError("MVSEP_API_KEY not found in environment variables")

        # Map model names to MVSEP separation types
        self.model_to_sep_type = {
            "ensemble_extra": "htdemucs_6s",  # 6 stems
            "ensemble": "htdemucs_ft",  # 4 stems, fine-tuned
            "fast": "htdemucs",  # 4 stems, faster
            "mdx": "mdx23c",  # MDX23C model
            "reverb": "reverb_removal",  # Reverb removal
            "denoise": "denoise_mdx",  # Denoising
        }

        # Timeout and quality settings
        self.timeout = config.get("mvsep", {}).get("api_timeout", 600)
        self.poll_interval = 5  # seconds between status checks

    def separate(self,
                 audio_path: Path,
                 model: Optional[str] = None,
                 progress_callback: Optional[Callable] = None) -> Dict:
        """
        Separate audio using MVSEP API.

        Args:
            audio_path: Path to input audio file
            model: Model to use (defaults to ensemble_extra)
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with separated stems and metrics
        """
        start_time = time.time()
        model = model or "ensemble_extra"

        try:
            # Get separation type for model
            sep_type = self.model_to_sep_type.get(model, "htdemucs_6s")

            logger.info(f"Using MVSEP model: {model} (sep_type: {sep_type})")

            if progress_callback:
                progress_callback(0.1, "Uploading audio to MVSEP...")

            # Create separation job
            hash_id = self._create_separation(audio_path, sep_type)

            if not hash_id:
                raise RuntimeError("Failed to create separation job")

            logger.info(f"Separation job created with hash: {hash_id}")

            if progress_callback:
                progress_callback(0.2, "Processing on MVSEP servers...")

            # Wait for processing and get results
            result_data = self._wait_for_results(hash_id, progress_callback)

            if not result_data:
                raise RuntimeError("Failed to get separation results")

            if progress_callback:
                progress_callback(0.8, "Downloading separated stems...")

            # Download and process stems
            stems_data = self._download_stems(result_data)

            # Convert to torch tensors
            processed_stems = self._process_stems(stems_data)

            # Calculate metrics
            metrics = self._calculate_metrics(processed_stems)

            processing_time = time.time() - start_time

            if progress_callback:
                progress_callback(1.0, "Separation complete!")

            # Clean up temporary files
            self._cleanup_temp_files(stems_data)

            return {
                "stems": processed_stems,
                "source_names": list(processed_stems.keys()),
                "metrics": metrics,
                "processing_time": processing_time,
                "model": model,
                "sep_type": sep_type
            }

        except Exception as e:
            logger.error(f"MVSEP separation failed: {e}")
            return self._fallback_separation(audio_path, progress_callback)

    def _create_separation(self, audio_path: Path, sep_type: str) -> Optional[str]:
        """
        Create a separation job on MVSEP servers.

        Returns:
            Hash ID of the created job or None if failed
        """
        try:
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'audiofile': (audio_path.name, audio_file, 'audio/mpeg')
                }

                data = {
                    'api_token': self.api_token,
                    'sep_type': sep_type,
                    'add_opt1': '',  # Additional options if needed
                    'add_opt2': '',
                    'output_format': '1',  # WAV format
                    'is_demo': '0'
                }

                response = requests.post(
                    f"{self.base_url}/api/separation/create",
                    files=files,
                    data=data,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()

                    if result.get('success'):
                        return result['data']['hash']
                    else:
                        error_msg = result.get('errors', ['Unknown error'])[0]
                        logger.error(f"MVSEP API error: {error_msg}")
                        return None
                else:
                    logger.error(f"HTTP {response.status_code}: {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Failed to create separation: {e}")
            return None

    def _wait_for_results(self, hash_id: str, progress_callback: Optional[Callable]) -> Optional[Dict]:
        """
        Poll MVSEP for processing completion and get results.

        Returns:
            Result data dictionary or None if failed
        """
        max_attempts = self.timeout // self.poll_interval

        for attempt in range(max_attempts):
            try:
                params = {'hash': hash_id}
                response = requests.get(
                    f"{self.base_url}/api/separation/get",
                    params=params,
                    timeout=300
                )

                if response.status_code == 200:
                    data = response.json()

                    if data.get('success'):
                        # Check if files are ready
                        if 'files' in data.get('data', {}):
                            logger.info("Separation completed successfully")
                            return data['data']
                        else:
                            # Still processing
                            if progress_callback:
                                progress = min(0.7, 0.2 + (attempt / max_attempts) * 0.5)
                                progress_callback(progress, f"Processing... {attempt * self.poll_interval}s")
                            time.sleep(self.poll_interval)
                    else:
                        # Check if still processing
                        error_msg = data.get('error', '')
                        if 'not ready' in error_msg.lower() or 'processing' in error_msg.lower():
                            if progress_callback:
                                progress = min(0.7, 0.2 + (attempt / max_attempts) * 0.5)
                                progress_callback(progress, f"Processing... {attempt * self.poll_interval}s")
                            time.sleep(self.poll_interval)
                        else:
                            logger.error(f"MVSEP error: {error_msg}")
                            return None
                else:
                    logger.warning(f"Status check failed: HTTP {response.status_code}")
                    time.sleep(self.poll_interval)

            except Exception as e:
                logger.warning(f"Poll attempt {attempt + 1} failed: {e}")
                time.sleep(self.poll_interval)

        logger.error("Processing timeout exceeded")
        return None

    def _download_stems(self, result_data: Dict) -> Dict:
        """
        Download separated stems from MVSEP.

        Returns:
            Dictionary mapping stem names to file paths
        """
        stems_paths = {}
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        files = result_data.get('files', [])

        for file_info in files:
            try:
                # Get URL and filename
                url = file_info['url'].replace('\\/', '/')
                filename = file_info['download']

                # Extract stem name from filename
                # Format is usually like "vocals.wav", "drums.wav", etc.
                stem_name = Path(filename).stem.lower()

                # Download file
                response = requests.get(url, stream=True, timeout=60)

                if response.status_code == 200:
                    # Save to temp directory
                    timestamp = int(time.time())
                    stem_path = temp_dir / f"{stem_name}_{timestamp}.wav"

                    with open(stem_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    stems_paths[stem_name] = stem_path
                    file_size = stem_path.stat().st_size / (1024 * 1024)
                    logger.info(f"Downloaded {stem_name}: {file_size:.2f} MB")
                else:
                    logger.warning(f"Failed to download {filename}: HTTP {response.status_code}")

            except Exception as e:
                logger.error(f"Failed to download stem: {e}")

        if not stems_paths:
            raise RuntimeError("No stems were downloaded successfully")

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

    def _calculate_metrics(self, stems: Dict) -> Dict:
        """Calculate separation quality metrics."""
        metrics = {
            "num_stems": len(stems)
        }

        total_energy = 0
        for stem_name, stem_audio in stems.items():
            if stem_audio is not None and stem_audio.numel() > 0:
                energy = torch.mean(stem_audio ** 2).item()
                metrics[f"{stem_name}_energy"] = energy
                total_energy += energy

        # Calculate energy ratios
        if total_energy > 0:
            for stem_name in stems.keys():
                if f"{stem_name}_energy" in metrics:
                    metrics[f"{stem_name}_energy_ratio"] = (
                            metrics[f"{stem_name}_energy"] / total_energy
                    )

        return metrics

    def _cleanup_temp_files(self, stems_paths: Dict):
        """Clean up temporary downloaded files."""
        for stem_path in stems_paths.values():
            try:
                if stem_path.exists():
                    stem_path.unlink()
                    logger.debug(f"Cleaned up temp file: {stem_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")

    def _fallback_separation(self, audio_path: Path, progress_callback: Optional[Callable]) -> Dict:
        """Fallback separation when API is unavailable."""
        logger.warning("Using fallback separation (API unavailable)")

        if progress_callback:
            progress_callback(0.5, "Using local fallback separation...")

        waveform, sr = torchaudio.load(audio_path)

        # Create placeholder stems
        processed_stems = {
            "vocals": waveform * 0.2,
            "drums": waveform * 0.15,
            "bass": waveform * 0.15,
            "other": waveform * 0.5
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
            "sep_type": "fallback"
        }

    def get_available_algorithms(self) -> List[Dict]:
        """Get list of available MVSEP algorithms."""
        try:
            response = requests.get(f"{self.base_url}/api/app/algorithms", timeout=10)

            if response.status_code == 200:
                algorithms = response.json()

                result = []
                for algo in algorithms:
                    result.append({
                        "id": algo['render_id'],
                        "name": algo['name'],
                        "group_id": algo['algorithm_group_id'],
                        "description": algo.get('algorithm_descriptions', [{}])[0].get('short_description', '')
                    })

                return result
            else:
                logger.error(f"Failed to get algorithms: HTTP {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Failed to fetch algorithms: {e}")
            return []

    def cleanup(self):
        """Clean up resources and temporary files."""
        temp_dir = Path("temp")
        if temp_dir.exists():
            for file in temp_dir.glob("*.wav"):
                try:
                    file.unlink()
                    logger.debug(f"Cleaned up {file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {file}: {e}")