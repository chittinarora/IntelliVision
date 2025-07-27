# /apps/video_analytics/convert.py

"""
Utility for converting videos and images to web-friendly formats using ffmpeg and PIL.
"""

import logging
import subprocess
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {'.mp4', '.avi', '.mov', '.webm', '.jpg', '.jpeg', '.png'}


def convert_to_web_mp4(input_file: str, output_file: str) -> bool:
    """
    Convert video to web-friendly MP4 (H.264, yuv420p).

    Args:
        input_file: Path to input video
        output_file: Path to output MP4

    Returns:
        bool: True if successful, False otherwise
    """
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return False

    ext = Path(input_file).suffix.lower()
    if ext not in {'.mp4', '.avi', '.mov', '.webm'}:
        logger.error(f"Invalid video type: {ext}")
        return False

    command = [
        "ffmpeg", "-y", "-i", input_file,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", output_file
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Converted {input_file} -> {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion failed: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found in PATH")
        return False


def convert_to_web_image(input_file: str, output_file: str, format: str = "WEBP", quality: int = 80) -> bool:
    """
    Convert image to web-friendly format (WebP or JPEG).

    Args:
        input_file: Path to input image
        output_file: Path to output image
        format: Output format ('WEBP' or 'JPEG')
        quality: Output quality (0-100)

    Returns:
        bool: True if successful, False otherwise
    """
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return False

    ext = Path(input_file).suffix.lower()
    if ext not in {'.jpg', '.jpeg', '.png'}:
        logger.error(f"Invalid image type: {ext}")
        return False

    try:
        with Image.open(input_file) as img:
            img = img.convert("RGB")
            img.save(output_file, format=format, quality=quality, optimize=True)
        logger.info(f"Converted {input_file} -> {output_file}")
        return True
    except Exception as e:
        logger.error(f"Image conversion failed: {e}")
        return False
