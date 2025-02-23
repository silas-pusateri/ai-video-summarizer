import os
import subprocess


from ai_jobs import (
    create_media_clips,
    generate_content,
    get_transcription_result,
    start_transcription,
)
from s3 import get_s3_presigned_url, upload_to_s3
from log import logger, save_debug_info
from transcription_goal import TranscriptionGoal
from utils import load_config, prompt_for_goal, prompt_for_media_file
from exporters import get_exporter


# Generate FFmpeg commands
def execute_ffmpeg_commands(commands):
    logger.debug(f"Executing FFmpeg commands: {commands}")
    for command in commands.split("&&"):
        logger.debug(f"Executing command: {command.strip()}")
        subprocess.run(command.strip(), shell=True, check=True)
    logger.info("All FFmpeg commands executed successfully")


def main(
    media_file, 
    goal=TranscriptionGoal.GENERAL_TRANSCRIPTION, 
    progress_callback=None,
    runtime_config=None
):
    try:
        logger.info(f"Starting main process for file: {media_file}")
        if progress_callback:
            progress_callback("Starting transcription process", 0)

        config = load_config()
        
        # Merge runtime config if provided
        if runtime_config:
            config.update(runtime_config)
            
        logger.debug(f"Using configuration: {config}")

        # Get the configured exporter
        export_format = config.get("export_format")
        logger.debug(f"Export format from config: '{export_format}'")
        exporter = get_exporter(export_format)
        logger.debug(f"Using exporter for format: {export_format or 'markdown'}")

        if progress_callback:
            progress_callback("Uploading media to S3", 10)
        upload_to_s3(media_file, config)

        if progress_callback:
            progress_callback("Getting presigned URL", 20)
        presigned_url = get_s3_presigned_url(os.path.basename(media_file), config)

        if progress_callback:
            progress_callback("Starting transcription", 30)
        prediction = start_transcription(presigned_url, config)

        if progress_callback:
            progress_callback("Processing transcription", 40)
        transcript = get_transcription_result(prediction["urls"]["get"], config)

        if progress_callback:
            progress_callback(f"Generating {goal.value.replace('_', ' ')}", 60)
        content = generate_content(transcript, goal, config)

        output_name = os.path.splitext(os.path.basename(media_file))[0]
        output_folder = os.path.join(os.path.dirname(media_file), output_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Format transcription content
        transcription_content = ""
        for segment in transcript:
            transcription_content += f"{segment['start']} - {segment['end']}: {segment['text']}\n"
        
        # Save transcription using configured exporter
        transcription_file = os.path.join(output_folder, f"{output_name}_transcription{exporter.get_extension()}")
        with open(transcription_file, 'wb') as f:
            f.write(exporter.export(transcription_content))
        logger.info(f"Transcription saved to {transcription_file}")

        # Save content using configured exporter
        output_file = os.path.join(output_folder, f"{output_name}_{goal.value}{exporter.get_extension()}")
        logger.info(f"Writing content to file: {output_file}")
        with open(output_file, "wb") as f:
            f.write(exporter.export(content))

        if progress_callback:
            progress_callback("Creating media clips", 80)
        ffmpeg_commands, topics, clips = create_media_clips(
            transcript, content, media_file, output_folder, goal, config
        )

        # Save debug information
        save_debug_info(output_folder, content, topics, clips)

        execute_ffmpeg_commands(ffmpeg_commands)

        if progress_callback:
            progress_callback("Process complete", 100)

        logger.info("Main process completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logger.info("Script started")
    media_file = prompt_for_media_file()
    if media_file:
        logger.info(f"Media file selected: {media_file}")
        goal = prompt_for_goal()
        logger.info(f"Transcription goal selected: {goal.value}")
        main(media_file, goal)
    else:
        logger.warning("No media file selected. Exiting.")
