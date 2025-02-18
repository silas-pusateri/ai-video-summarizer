from typing import Dict, List
import requests
import time
from log import logger

class TranscriptionModel:
    """Base class for transcription models."""
    
    def start_transcription(self, url: str, config: Dict) -> Dict:
        """Initiate transcription and return the initial prediction object."""
        raise NotImplementedError

    def get_transcription_result(self, prediction_url: str, config: Dict) -> List[Dict]:
        """Poll for and return the transcription segments."""
        raise NotImplementedError


class ReplicateTranscriptionModel(TranscriptionModel):
    """Implementation of Replicate's transcription model."""

    def start_transcription(self, url: str, config: Dict) -> Dict:
        logger.debug(f"Starting transcription for URL: {url}")
        headers = {
            "Authorization": f"Bearer {config['replicate_api_key']}",
            "Content-Type": "application/json",
        }
        data = {
            "version": config["replicate_model_version"],
            "input": {
                "debug": True,
                "language": "en",
                "vad_onset": 0.5,
                "audio_file": url,
                "batch_size": 64,
                "vad_offset": 0.363,
                "diarization": False,
                "temperature": 0,
                "align_output": False,
                "huggingface_access_token": config["huggingface_token"],
                "language_detection_min_prob": 0,
                "language_detection_max_tries": 5,
            },
        }
        logger.debug(f"Sending request to Replicate API: {config['replicate_api_url']}")
        response = requests.post(config["replicate_api_url"], headers=headers, json=data)
        logger.debug(f"Replicate API response: {response.text}")
        return response.json()

    def get_transcription_result(self, prediction_url: str, config: Dict) -> List[Dict]:
        logger.debug(f"Getting transcription result from: {prediction_url}")
        headers = {"Authorization": f"Bearer {config['replicate_api_key']}"}
        while True:
            response = requests.get(prediction_url, headers=headers)
            result = response.json()
            logger.debug(f"Transcription status: {result['status']}")
            if result["status"] == "succeeded":
                logger.info("Transcription completed successfully")
                return result["output"]["segments"]
            elif result["status"] == "failed":
                logger.error("Transcription process failed")
                raise Exception("Transcription process failed.")
            time.sleep(5)


def get_transcription_model(config: Dict) -> TranscriptionModel:
    """Factory function to get the appropriate transcription model."""
    model_type = config.get("transcription_model", "replicate")
    if model_type == "replicate":
        return ReplicateTranscriptionModel()
    else:
        raise ValueError(f"Unknown transcription model: {model_type}") 