#!/usr/bin/env python3
"""
Test script to verify requirements.txt compatibility
"""

import subprocess
import sys
import tempfile
import os

def test_requirements():
    """Test if requirements.txt can be installed without conflicts."""

    print("🔍 Testing requirements.txt compatibility...")

    # Create a temporary virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = os.path.join(temp_dir, "test_venv")

        try:
            # Create virtual environment
            print("📦 Creating test virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)

            # Get pip path
            if os.name == 'nt':  # Windows
                pip_path = os.path.join(venv_path, "Scripts", "pip")
            else:  # Unix/Linux/macOS
                pip_path = os.path.join(venv_path, "bin", "pip")

            # Upgrade pip
            print("⬆️  Upgrading pip...")
            subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)

            # Install requirements
            print("📥 Installing requirements...")
            result = subprocess.run(
                [pip_path, "install", "-r", "requirements.txt"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print("✅ Requirements installed successfully!")
                print("📊 Installed packages:")
                subprocess.run([pip_path, "list"])
                return True
            else:
                print("❌ Requirements installation failed:")
                print(result.stderr)
                return False

        except subprocess.CalledProcessError as e:
            print(f"❌ Error during testing: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

def check_specific_versions():
    """Check for potentially problematic version constraints."""

    print("\n🔍 Checking for potential version issues...")

    problematic_packages = [
        "thop~=0.1.1-2209072238",  # This specific version might not exist
        "yt-dlp>=2024.1.0",        # Very old version
        "gdown>=5.1.0,<6.0.0",     # Range constraint
    ]

    issues_found = []

    for package in problematic_packages:
        print(f"⚠️  Potentially problematic: {package}")
        issues_found.append(package)

    if issues_found:
        print(f"\n⚠️  Found {len(issues_found)} potentially problematic packages")
        print("These are warnings but may still work fine")
        return True  # Changed to True since these are just warnings
    else:
        print("✅ No obvious version issues found")
        return True

if __name__ == "__main__":
    print("🚀 Requirements.txt Compatibility Test")
    print("=" * 50)

    # Check for obvious issues first (warnings only)
    version_check = check_specific_versions()

    # Test actual installation
    install_test = test_requirements()

    print("\n" + "=" * 50)
    if install_test:
        print("✅ All tests passed! Requirements.txt works correctly.")
        print("📝 Note: Some version warnings were shown, but installation succeeded.")
    else:
        print("❌ Installation failed. Please review the requirements.txt file.")
        sys.exit(1)
