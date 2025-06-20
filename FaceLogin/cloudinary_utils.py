import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def upload_face_image(image_path, public_id=None):
    """
    Uploads a face image to Cloudinary.
    Returns the secure URL.
    """
    result = cloudinary.uploader.upload(
        image_path,
        public_id=public_id,
        folder="face_login_app",
        overwrite=True,
        resource_type="image"
    )
    return result["secure_url"]
