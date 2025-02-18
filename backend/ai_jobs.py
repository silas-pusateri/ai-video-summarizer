import json
import time
import os
import re
import subprocess

import requests

from log import logger
from transcription_goal import TranscriptionGoal


def start_transcription(url, config):
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


def get_transcription_result(prediction_url, config):
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


def generate_content(transcript, goal, config):
    logger.debug(f"Generating content for goal: {goal.value}")
    headers = {
        "x-api-key": config["anthropic_api_key"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    prompts = {
        TranscriptionGoal.MEETING_MINUTES: "Create very detailed meeting minutes based on the following transcription. Return the response as a JSON object with a 'content' field containing the meeting minutes:",
        TranscriptionGoal.PODCAST_SUMMARY: "Summarize this podcast episode, highlighting key points and interesting discussions. Return the response as a JSON object with a 'content' field containing the summary:",
        TranscriptionGoal.LECTURE_NOTES: "Create comprehensive lecture notes from this transcription, organizing key concepts and examples. Return the response as a JSON object with a 'content' field containing the lecture notes:",
        TranscriptionGoal.INTERVIEW_HIGHLIGHTS: "Extract the main insights and notable quotes from this interview transcription. Return the response as a JSON object with a 'content' field containing the highlights:",
        TranscriptionGoal.GENERAL_TRANSCRIPTION: "Provide a clear and concise summary of the main points discussed in this transcription. Return the response as a JSON object with a 'content' field containing the summary:",
    }

    prompt = prompts.get(goal, prompts[TranscriptionGoal.GENERAL_TRANSCRIPTION])
    
    # Updated request format for Anthropic's messages API
    data = {
        "model": config["anthropic_model"],
        "messages": [
            {
                "role": "user",
                "content": f"""You are a specialized content generation assistant. You must ALWAYS return responses in valid JSON format.
                Do not include any explanatory text, markdown, or other formatting outside the JSON structure.
                If you need to include newlines in the content, use '\n' in the JSON string.

                {prompt} {json.dumps(transcript)}"""
            }
        ],
        "max_tokens": 4000,
        "temperature": 0
    }
    
    logger.debug(f"Sending request to Anthropic API: {config['anthropic_api_url']}")
    response = requests.post(config["anthropic_api_url"], headers=headers, json=data)
    logger.debug(f"Anthropic API response: {response.text}")

    try:
        response_json = response.json()
        
        # Check for API errors first
        if "error" in response_json:
            error_msg = response_json.get("error", {}).get("message", "Unknown error")
            logger.error(f"Anthropic API error: {error_msg}")
            raise Exception(f"Anthropic API error: {error_msg}")
            
        # Get the text content from the response
        message_content = response_json.get("content", [{}])[0].get("text", "")
        if not message_content:
            raise Exception("No content in response")
            
        # Clean the content string of any control characters
        message_content = "".join(char for char in message_content if ord(char) >= 32 or char == '\n')
            
        try:
            # First try to parse the entire message content
            content = json.loads(message_content)
            return content["content"]
        except (json.JSONDecodeError, KeyError):
            # If that fails, try to extract just the content field using regex
            import re
            content_match = re.search(r'"content"\s*:\s*"([^"]*(?:"[^"]*"[^"]*)*)"', message_content)
            if content_match:
                content_text = content_match.group(1)
                # Unescape any escaped quotes within the content
                content_text = content_text.replace('\\"', '"')
                return content_text
            else:
                logger.error(f"Could not extract content from response: {message_content}")
                raise Exception("Could not parse content from AI response")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response as JSON: {e}")
        logger.error(f"Raw response content: {message_content}")
        raise Exception("AI response was not in valid JSON format")
    except KeyError as e:
        logger.error(f"Unexpected response structure: {e}")
        logger.error(f"Response JSON: {response_json}")
        raise Exception("Unexpected response structure from Anthropic API")
    except Exception as e:
        logger.error(f"Unexpected error processing AI response: {str(e)}")
        raise Exception(f"Error processing AI response: {str(e)}")


def create_media_clips(transcript, content, source_file, dest_folder, goal, config):
    logger.debug(f"Creating media clips for goal: {goal.value}")
    
    topic_extraction_message = f"""You are a specialized JSON generation assistant. You must ALWAYS return responses in valid JSON format.
    Do not include any explanatory text, markdown, or other formatting outside the JSON array structure.

    Based on the following {goal.value.replace('_', ' ')}:
    {content}

    Extract each of the main topics or segments discussed.
    Return ONLY a JSON array of objects, with each object containing:
    - 'title': A short, descriptive title (max 5 words)
    - 'keywords': An array of related keywords (max 5 keywords)

    Do not include any explanatory text or formatting outside the JSON structure.
    """

    headers = {
        "x-api-key": config['anthropic_api_key'],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    topic_extraction_data = {
        "model": config['anthropic_model'],
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": topic_extraction_message
            }
        ]
    }

    topic_response = requests.post(config['anthropic_api_url'], headers=headers, json=topic_extraction_data)
    topic_text = topic_response.json()['content'][0]['text']
    logger.debug(f"Full AI response for topic extraction:\n{topic_text}")

    try:
        topics = json.loads(topic_text)
    except json.JSONDecodeError:
        topic_pattern = r'\{\s*"title":\s*"([^"]+)",\s*"keywords":\s*\[((?:[^]]+))\]\s*\}'  
        matches = re.findall(topic_pattern, topic_text)
        topics = [{"title": title, "keywords": [k.strip(' "') for k in keywords.split(',')]} for title, keywords in matches]

    if not topics:
        raise ValueError("Failed to extract topics from the AI response")

    clip_generation_message = f"""
    For each of the following topics/segments, find the most relevant part in the transcript:
    {json.dumps(topics)}

    Transcript:
    {json.dumps(transcript)}

    Return ONLY a JSON array of objects, with each object containing:
    - 'title': The topic/segment title
    - 'start': Start time of the clip (in seconds)
    - 'end': End time of the clip (in seconds)

    Rules for clip selection:
    1. Aim for 2-5 minutes duration
    2. Include complete discussions even if over 5 minutes
    3. Never cut off mid-sentence
    4. Clips can overlap if needed

    Do not include any explanatory text or formatting outside the JSON structure.
    """

    clip_generation_data = {
        "model": config['anthropic_model'],
        "max_tokens": 2000,
        "messages": [
            {"role": "user", "content": clip_generation_message}
        ]
    }

    clip_response = requests.post(config['anthropic_api_url'], headers=headers, json=clip_generation_data)
    clip_text = clip_response.json()['content'][0]['text']
    logger.debug(f"Full AI response for clip generation:\n{clip_text}")

    try:
        clips = json.loads(clip_text)
    except json.JSONDecodeError:
        clip_pattern = r'\{\s*"title":\s*"([^"]+)",\s*"start":\s*(\d+(?:\.\d+)?),\s*"end":\s*(\d+(?:\.\d+)?)\s*\}'
        matches = re.findall(clip_pattern, clip_text)
        clips = [{"title": title, "start": float(start), "end": float(end)} for title, start, end in matches]

    if not clips:
        raise ValueError("Failed to extract clip information from the AI response")

    # Find ffmpeg path or use default
    try:
        ffmpeg_path = subprocess.check_output(['which', 'ffmpeg'], text=True).strip()
    except subprocess.CalledProcessError:
        # If which command fails, try common locations
        for path in ['/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/opt/homebrew/bin/ffmpeg']:
            if os.path.exists(path):
                ffmpeg_path = path
                break
        else:
            raise Exception("ffmpeg not found. Please install ffmpeg and ensure it's in your PATH")

    ffmpeg_commands = []
    for clip in clips:
        safe_title = ''.join(c for c in clip['title'] if c.isalnum() or c in (' ', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')
        output_file = os.path.join(dest_folder, f"{safe_title}{os.path.splitext(source_file)[1]}")
        start_time = clip['start']
        end_time = clip['end']
        buffer = 0.5
        start_time = max(0, start_time - buffer)
        end_time += buffer
        command = f'"{ffmpeg_path}" -i "{source_file}" -ss {start_time:.2f} -to {end_time:.2f} -y -c copy "{output_file}"'
        ffmpeg_commands.append(command)

    logger.debug(f"Generated FFmpeg commands: {ffmpeg_commands}")
    return ' && '.join(ffmpeg_commands), topics, clips