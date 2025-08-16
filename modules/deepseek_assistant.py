import requests
from typing import Dict, Optional
import logging
import json

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

    def explain_audiocraft_processing(self, config: Dict, metrics: Dict) -> str:
        """Explain AudioCraft processing."""
        prompt = f"""
        Explain the AudioCraft processing with model {config['model']}:
        - Temperature: {config['temperature']}
        - Duration: {metrics.get('duration', 0):.2f} seconds

        What do these parameters control and how do they affect the output?
        """

        response = self._query(prompt)
        return response or "AudioCraft processing applied successfully."

    def explain_style_transfer(self, config: Dict, metrics: Dict) -> str:
        """Explain DDSP style transfer."""
        prompt = f"""
        Explain DDSP style transfer to {config['checkpoint']} instrument:
        - Pitch shift: {config['pitch_shift']} semitones
        - Loudness shift: {config['loudness_shift']} dB

        How does DDSP achieve timbre transfer?
        """

        response = self._query(prompt)
        return response or f"Style transfer to {config['checkpoint']} completed."

    def classify_instruments(self, features: Dict) -> Dict:
        """Classify instruments and suggest MVSEP separation strategy."""
        prompt = f"""
        Analyze these audio features and recommend the best MVSEP model:
    
        Audio Features:
        - Harmonic ratio: {features.get('harmonic_ratio', 0):.2f}
        - Spectral centroid: {features.get('spectral_centroid_mean', 0):.1f} Hz
        - Tempo: {features.get('tempo_bpm', 0):.1f} BPM
        - Bass energy: {features.get('bass_energy', 0):.2f}
        - Mid energy: {features.get('mid_energy', 0):.2f}
        - High energy: {features.get('high_energy', 0):.2f}
    
        AVAILABLE MVSEP MODELS (choose ONE):
        - "ensemble": Best overall quality, 4 stems (vocals, drums, bass, other)
        - "ensemble_extra": 6 stems (vocals, drums, bass, piano, guitar, other)
        - "vocal_instrumental": 2 stems (vocals, instrumental) - for karaoke
        - "karaoke": 2 stems - optimized for vocal removal
        - "drums_focus": 4 stems - enhanced drum separation
        - "piano": 2 stems - specialized for piano isolation
        - "guitar": 2 stems - specialized for guitar isolation
        - "strings": 2 stems - specialized for string instruments
        - "fast": 4 stems - faster processing
        - "ultra_fast": 4 stems - fastest processing
    
        Based on the audio features, return ONLY this JSON structure:
        {{
            "recommended_model": "ensemble",
            "confidence": 0.8,
            "reasoning": "Brief reason for this choice",
            "expected_stems": 4
        }}
        """

        response = self._query(prompt)
        # Parse JSON response
        try:
            if not response:
                return {}

            try:
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:]  # Remove ```
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]  # Remove ending ```
                cleaned_response = cleaned_response.strip()

                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response from DeepSeek: {response[:100]}...")
                return {}
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response from DeepSeek")
            return {
                "instruments": [],
                "separation_strategy": "Use default ensemble model"
            }