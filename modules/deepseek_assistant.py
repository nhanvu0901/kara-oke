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
        """Use LLM to classify instruments and recommend separation strategy."""

        prompt = f"""
        Analyze these audio features and identify likely instruments present:

        Harmonic Ratio: {features.get('harmonic_ratio', 0):.3f}
        Percussive Ratio: {features.get('percussive_ratio', 0):.3f}
        Spectral Complexity: {features.get('spectral_complexity', 0):.3f}
        Attack Sharpness: {features.get('attack_sharpness', 0):.3f}
        Vocal Presence: {features.get('vocal_presence', 0):.3f}
        Onset Density: {features.get('onset_density', 0):.2f}

        Frequency Band Energy Distribution:
        - Sub-bass (20-250Hz): {features.get('band_0_energy', 0):.0f}
        - Bass (250-500Hz): {features.get('band_1_energy', 0):.0f}
        - Midrange (500-2kHz): {features.get('band_2_energy', 0):.0f}
        - Upper-mid (2-4kHz): {features.get('band_3_energy', 0):.0f}
        - Presence (4-8kHz): {features.get('band_4_energy', 0):.0f}
        - Brilliance (8-20kHz): {features.get('band_5_energy', 0):.0f}

        Based on these features:
        1. List the likely instruments present (with confidence 0-1)
        2. Recommend optimal separation models for these instruments
        3. Suggest separation strategy (single-pass, multi-pass, ensemble)
        4. Identify potential challenges for separation

        Respond in JSON format:
        {{
            "instruments": {{"instrument_name": confidence}},
            "recommended_models": ["model1", "model2"],
            "strategy": {{"passes": N, "method": "...", "priority": "..."}},
            "challenges": ["challenge1", "challenge2"]
        }}
        """

        response = self._query(prompt)

        if response:
            try:
                import json
                return json.loads(response)
            except:
                return {
                    "instruments": {"unknown": 0.5},
                    "recommended_models": ["htdemucs_ft"],
                    "strategy": {"passes": 1, "method": "standard", "priority": "balanced"},
                    "challenges": ["Unable to parse LLM response"]
                }

        return {
            "instruments": {"unknown": 0.5},
            "recommended_models": ["htdemucs_ft"],
            "strategy": {"passes": 1, "method": "standard", "priority": "balanced"},
            "challenges": ["No LLM available"]
        }
