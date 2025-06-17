import os
import cloudinary
import cloudinary.uploader

# No need for load_dotenv here; it's loaded in settings.py for the whole project

# Configure Cloudinary with environment variables
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
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
