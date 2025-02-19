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


class ReplicateBaseModel:
    """Base class for Replicate-hosted models with common functionality."""
    
    def __init__(self, model_version: str):
        """Initialize with specific model version."""
        self.model_version = model_version

    def _make_replicate_request(self, url: str, input_data: Dict, config: Dict) -> Dict:
        """Make a request to Replicate API with the given input data."""
        headers = {
            "Authorization": f"Bearer {config['replicate_api_key']}",
            "Content-Type": "application/json",
        }
        data = {
            "version": self.model_version,
            "input": input_data,
        }
        logger.debug(f"Sending request to Replicate API: {config['replicate_api_url']}")
        response = requests.post(config["replicate_api_url"], headers=headers, json=data)
        logger.debug(f"Replicate API response: {response.text}")
        response.raise_for_status()
        return response.json()

    def _poll_replicate_result(self, prediction_url: str, config: Dict) -> Dict:
        """Poll Replicate API for results."""
        headers = {"Authorization": f"Bearer {config['replicate_api_key']}"}
        while True:
            response = requests.get(prediction_url, headers=headers)
            result = response.json()
            logger.debug(f"Transcription status: {result['status']}")
            if result["status"] == "succeeded":
                logger.info("Transcription completed successfully")
                return result
            elif result["status"] == "failed":
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Transcription process failed: {error_msg}")
                raise Exception(f"Transcription process failed: {error_msg}")
            time.sleep(5)


class WhisperXTranscriptionModel(TranscriptionModel, ReplicateBaseModel):
    """Implementation of Replicate's WhisperX transcription model."""

    def start_transcription(self, url: str, config: Dict) -> Dict:
        logger.debug(f"Starting transcription for URL: {url} using WhisperX")
        input_data = {
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
        }
        return self._make_replicate_request(url, input_data, config)

    def get_transcription_result(self, prediction_url: str, config: Dict) -> List[Dict]:
        result = self._poll_replicate_result(prediction_url, config)
        return result["output"]["segments"]


class IncrediblyFastWhisperTranscriptionModel(TranscriptionModel, ReplicateBaseModel):
    """Implementation of Replicate's incredibly-fast-whisper transcription model."""

    def start_transcription(self, url: str, config: Dict) -> Dict:
        logger.debug(f"Starting transcription for URL: {url} using incredibly-fast-whisper")
        input_data = {
            "audio": url,  # Different key from WhisperX
            "language": "english",  # Must be full language name, not code
            "task": "transcribe",
            "timestamp_type": "absolute",  # Control timestamp format
            "word_timestamps": "true",  # Must be string "true"/"false", not boolean
            "return_segments": "true",  # Must be string "true"/"false", not boolean
        }
        return self._make_replicate_request(url, input_data, config)

    def get_transcription_result(self, prediction_url: str, config: Dict) -> List[Dict]:
        result = self._poll_replicate_result(prediction_url, config)
        
        # incredibly-fast-whisper returns segments in a different format
        # Convert it to match the expected format (same as WhisperX)
        try:
            raw_segments = result["output"].get("segments", [])
            segments = []
            
            for segment in raw_segments:
                # Extract required fields and ensure consistent format
                segments.append({
                    "start": float(segment.get("start", 0)),
                    "end": float(segment.get("end", 0)),
                    "text": segment.get("text", "").strip(),
                })
            
            logger.info(f"Successfully converted {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to parse incredibly-fast-whisper output: {str(e)}")
            logger.error(f"Raw output was: {result}")
            raise Exception(f"Failed to parse transcription result: {str(e)}")


class LocalWhisperTranscriptionModel(TranscriptionModel):
    """Implementation of local Whisper model."""

    def __init__(self, model_path: str):
        """Initialize with path to local model."""
        self.model_path = model_path
        # We'll load the model lazily when needed to avoid unnecessary memory usage
        self._model = None

    def _ensure_model_loaded(self):
        """Load the model if not already loaded."""
        if self._model is None:
            try:
                import whisper
                logger.info(f"Loading Whisper model from {self.model_path}")
                self._model = whisper.load_model(self.model_path)
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {str(e)}")
                raise

    def start_transcription(self, url: str, config: Dict) -> Dict:
        """
        Start transcription using local Whisper model.
        For local processing, we'll return a dict that mimics Replicate's response format
        but contains the local file path instead of a prediction URL.
        """
        logger.debug(f"Starting local transcription for URL: {url}")
        return {
            "status": "processing",
            "local_file": url,  # We'll use this in get_transcription_result
        }

    def get_transcription_result(self, prediction_url: str, config: Dict) -> List[Dict]:
        """
        Process the audio file using local Whisper model.
        prediction_url in this case will be the local file path from start_transcription.
        """
        try:
            self._ensure_model_loaded()
            
            # Load and transcribe the audio
            logger.debug("Starting Whisper transcription")
            result = self._model.transcribe(prediction_url)
            
            # Convert Whisper output format to match Replicate's format
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                })
            
            logger.info("Local transcription completed successfully")
            return segments
            
        except Exception as e:
            logger.error(f"Local transcription failed: {str(e)}")
            raise Exception(f"Local transcription failed: {str(e)}")


def get_transcription_model(config: Dict) -> TranscriptionModel:
    """Factory function to get the appropriate transcription model."""
    model_type = config.get("transcription_model", "replicate")
    
    if model_type == "replicate":
        # Get the selected model version from the config
        available_versions = config.get("replicate_model_versions", {})
        selected = config.get("selected_replicate_model", "whisperx")
        model_version = available_versions.get(selected)
        
        if not model_version:
            raise ValueError(f"No model version found for selected Replicate model: {selected}")
        
        # Return the appropriate Replicate model based on selection
        if selected == "incredibly-fast-whisper":
            return IncrediblyFastWhisperTranscriptionModel(model_version)
        elif selected == "whisperx":
            return WhisperXTranscriptionModel(model_version)
        else:
            raise ValueError(f"Unknown Replicate model type: {selected}")
        
    elif model_type == "local_whisper":
        model_path = config.get("local_whisper_model_path")
        if not model_path:
            raise ValueError("local_whisper_model_path must be specified when using local_whisper model")
        return LocalWhisperTranscriptionModel(model_path)
        
    else:
        raise ValueError(f"Unknown transcription model: {model_type}") 