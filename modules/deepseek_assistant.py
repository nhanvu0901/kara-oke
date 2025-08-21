import requests
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DeepSeekAssistant:
    """AI assistant for educational explanations using DeepSeek."""

    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config["deepseek"]["api_key"]
        self.model = config["deepseek"]["model"]
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    def _query(self, prompt: str) -> Optional[str]:
        """Query DeepSeek API."""
        if not self.api_key:
            return None

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an educational AI audio processing assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.config["deepseek"]["temperature"],
                "max_tokens": self.config["deepseek"]["max_tokens"]
            }

            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()

            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return None

    def explain_audio_features(self, features: Dict) -> str:
        """Generate educational explanation of audio features."""
        prompt = f"""
        Explain these audio features in an educational way:
        - Duration: {features.get('duration', 0):.2f} seconds
        - Tempo: {features.get('tempo_bpm', 0):.1f} BPM
        - Spectral Centroid: {features.get('spectral_centroid_mean', 0):.2f} Hz

        Make it educational and easy to understand for someone learning audio processing.
        """

        response = self._query(prompt)
        return response or "Audio features extracted successfully."

    def explain_source_separation(self, metrics: Dict, model: str) -> str:
        """Explain source separation results."""
        prompt = f"""
        Explain the source separation results from {model}:
        - SNR: {metrics.get('reconstruction_snr', 0):.2f} dB
        - Number of stems: {metrics.get('num_stems', 0)}

        Explain what these metrics mean and why source separation is important.
        """

        response = self._query(prompt)
        return response or f"Source separation completed using {model}."
