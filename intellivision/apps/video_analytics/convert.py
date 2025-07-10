"""
convert.py - Video Analytics App
Utility for converting videos to web-friendly MP4 format using ffmpeg.
"""

import subprocess
import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)

def convert_to_web_mp4(input_file: str, output_file: str) -> bool:
    """
    Converts an input video file to a web-friendly MP4 format (H.264, yuv420p).

    Args:
        input_file (str): Path to the input video file.
        output_file (str): Path where the converted MP4 will be saved.

    Returns:
        bool: True if conversion is successful, False otherwise.
    """
    command = [
        "ffmpeg", "-y", "-i", input_file,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", output_file
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Converted {input_file} -> {output_file} (web compatible)")
        return True
    except subprocess.CalledProcessError as e:
        # ffmpeg command failed
        logger.error(f"ffmpeg conversion failed: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        # ffmpeg not installed or not in PATH
        logger.error("Error: ffmpeg not found. Please ensure it is installed and in your system's PATH.")
        return False
