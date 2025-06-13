import torchreid
import os

# The directory where weights are typically saved by default
# On Linux/macOS: ~/.cache/torch/checkpoints/
# On Windows: C:/Users/YourUsername/.cache/torch/checkpoints/
weights_dir = os.path.join(os.path.expanduser('~'), '.cache/torch/checkpoints')
os.makedirs(weights_dir, exist_ok=True) # Ensure the directory exists

print("Attempting to download weights for osnet_x1_0 on MSMT17...")
print(f"Weights will be saved to: {weights_dir}")

try:
    # This is the line that triggers the download.
    # It will build the model and fetch the pretrained weights from the source.
    torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=2510,  # Number of identities in MSMT17 dataset
        pretrained=True,   # This flag tells the library to download the weights
        use_gpu=False
    )
    print("\nSUCCESS: Model built and weights downloaded successfully!")
    print(f"Please check for the 'osnet_x1_0_msmt17.pt' file in the directory above.")

except Exception as e:
    print(f"\nAn error occurred during download: {e}")
    print("This might be due to a network issue or firewall.")
    print("Please try Method 2 if the problem persists.")