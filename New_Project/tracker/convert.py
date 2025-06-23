import subprocess


def convert_to_web_mp4(input_file: str, output_file: str) -> bool:
    """
    Converts an input video file to a web-friendly MP4 format (H.264, yuv420p).
    Returns True if successful, False otherwise.
    """
    command = [
        "ffmpeg", "-y", "-i", input_file,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", output_file
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Converted {input_file} -> {output_file} (web compatible)")
        return True
    except subprocess.CalledProcessError as e:
        # ffmpeg command failed
        print(f"ffmpeg conversion failed: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        # ffmpeg not installed or not in PATH
        print("Error: ffmpeg not found. Please ensure it is installed and in your system's PATH.")
        return False
