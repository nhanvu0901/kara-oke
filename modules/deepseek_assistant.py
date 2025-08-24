import requests
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DeepSeekAssistant:
    """AI assistant for educational explanations of instrumental separation using DeepSeek."""

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
                    {"role": "system", "content": "You are an educational AI assistant specializing in instrumental audio processing and stem separation. Focus on explaining bass, drums, and other instrumental components."},
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
        """Generate educational explanation of audio features for instrumental separation."""
        prompt = f"""
        Explain these audio features in the context of instrumental separation:
        - Duration: {features.get('duration', 0):.2f} seconds
        - Tempo: {features.get('tempo_bpm', 0):.1f} BPM
        - Spectral Centroid: {features.get('spectral_centroid_mean', 0):.2f} Hz
        - Harmonic/Percussive Ratio: {features.get('harmonic_percussive_ratio', 0):.2f}

        Focus on how these features relate to separating bass, drums, and other instrumental elements.
        Make it educational and explain why these metrics matter for instrumental separation.
        """

        response = self._query(prompt)
        return response or "Audio features analyzed for instrumental content."

    def explain_source_separation(self, metrics: Dict, model: str) -> str:
        """Explain instrumental source separation results."""
        prompt = f"""
        Explain the instrumental separation results from {model}:
        - Instrumental Coverage: {metrics.get('instrumental_coverage_ratio', 0):.1%}
        - Number of instrumental stems: {metrics.get('num_instrumental_stems', 0)}
        - Bass energy: {metrics.get('bass_energy', 0):.4f}
        - Drums energy: {metrics.get('drums_energy', 0):.4f}
        - Other instruments energy: {metrics.get('other_energy', 0):.4f}

        Explain what these metrics mean for instrumental separation quality.
        Focus on bass, drums, and other instrumental stems.
        Discuss applications like remixing, practice tracks, and music production.
        """

        response = self._query(prompt)
        return response or f"Instrumental separation completed using {model}."

    def explain_instrumental_stem(self, stem_name: str, analysis: Dict) -> str:
        """Explain characteristics of a specific instrumental stem."""
        prompt = f"""
        Explain the characteristics of the {stem_name} stem:
        - Energy: {analysis.get('rms_energy', 0):.4f}
        - Dynamic Range: {analysis.get('dynamic_range_db', 0):.1f} dB
        - Spectral Centroid: {analysis.get('spectral_centroid_mean', 0):.2f} Hz

        Provide educational insights about:
        1. What instruments typically appear in the {stem_name} stem
        2. Frequency ranges important for {stem_name}
        3. How to use this stem in music production or practice
        """

        response = self._query(prompt)
        return response or f"Analysis of {stem_name} instrumental stem completed."

    def suggest_instrumental_applications(self, stems: list) -> str:
        """Suggest applications for the separated instrumental stems."""
        prompt = f"""
        Given these separated instrumental stems: {', '.join(stems)}

        Suggest creative applications for these instrumental stems:
        1. Music production techniques
        2. Practice and learning applications
        3. Remixing possibilities
        4. Audio restoration or enhancement

        Focus on practical, educational uses for musicians and producers.
        """

        response = self._query(prompt)
        return response or "Instrumental stems can be used for remixing, practice, and production."