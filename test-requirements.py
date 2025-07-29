#!/usr/bin/env python3
"""
Test script to verify requirements.txt compatibility and installation
"""
import subprocess
import sys
import tempfile
import os
import re
import time
from pathlib import Path


def validate_requirements_file(filename="requirements.txt"):
    """Validate requirements.txt exists and has valid format."""
    print(f"📋 Validating {filename}...")

    if not os.path.exists(filename):
        print(f"❌ {filename} not found!")
        return False

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            print(f"⚠️  {filename} is empty")
            return False

        # Filter out comments and empty lines for validation
        package_lines = [line.strip() for line in lines
                        if line.strip() and not line.strip().startswith('#')]

        if not package_lines:
            print(f"⚠️  {filename} contains no packages (only comments/empty lines)")
            return False

        # Check for basic syntax issues
        for i, original_line in enumerate(lines, 1):
            line = original_line.strip()
            if line and not line.startswith('#'):
                # Basic package name validation - more permissive regex
                if not re.match(r'^[a-zA-Z0-9\-_\.]+(\[.*\])?([<>=!~,\s\d\.\-\w]*)?$', line):
                    print(f"❌ Potentially invalid package format on line {i}: {line}")
                    print("   (This might still work, but could cause issues)")

        print(f"✅ {filename} format appears valid ({len(package_lines)} packages found)")
        return True

    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return False


def check_version_patterns(filename="requirements.txt"):
    """Check for potentially problematic version patterns."""
    print(f"🔍 Analyzing version patterns in {filename}...")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
        return []

    issues = []
    warnings = []

    for line_num, line in enumerate(lines, 1):
        original_line = line
        line = line.strip()

        if not line or line.startswith('#'):
            continue

        # Check for potentially problematic patterns

        # 1. Unusual version formats with timestamps/dates
        if '~=' in line and '-' in line.split('~=')[1]:
            warnings.append(f"Line {line_num}: Unusual version format - {line}")

        # 2. Very specific date-based versions that might not exist
        if re.search(r'\d{8,}', line):  # 8 or more consecutive digits
            warnings.append(f"Line {line_num}: Date-based version detected - {line}")

        # 3. Complex version constraints
        version_operators = line.count('>=') + line.count('<=') + line.count('==') + line.count('!=') + line.count('>')+ line.count('<')
        if version_operators > 2:
            warnings.append(f"Line {line_num}: Complex version constraint - {line}")

        # 4. Pre-release versions
        if re.search(r'(alpha|beta|rc|dev|pre)', line.lower()):
            warnings.append(f"Line {line_num}: Pre-release version - {line}")

        # 5. Git/URL dependencies
        if any(prefix in line.lower() for prefix in ['git+', 'http://', 'https://', 'ftp://']):
            warnings.append(f"Line {line_num}: URL/Git dependency - {line}")

        # 6. Very old version constraints (might be outdated)
        if '>=2020' in line or '>=2019' in line or '>=2018' in line:
            warnings.append(f"Line {line_num}: Very old minimum version - {line}")

    if warnings:
        print(f"⚠️  Found {len(warnings)} potential issues (warnings):")
        for warning in warnings[:10]:  # Show first 10 warnings
            print(f"   • {warning}")
        if len(warnings) > 10:
            print(f"   ... and {len(warnings) - 10} more warnings")
        print("   Note: These are warnings and may still install successfully")
    else:
        print("✅ No obvious version pattern issues found")

    return warnings


def test_requirements(filename="requirements.txt"):
    """Test if requirements.txt can be installed without conflicts."""
    print(f"🔍 Testing {filename} installation compatibility...")

    # Create a temporary virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = os.path.join(temp_dir, "test_venv")

        try:
            # Create virtual environment
            print("📦 Creating test virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", venv_path],
                         check=True, capture_output=True)

            # Get pip path
            if os.name == 'nt':  # Windows
                pip_path = os.path.join(venv_path, "Scripts", "pip")
                python_path = os.path.join(venv_path, "Scripts", "python")
            else:  # Unix/Linux/macOS
                pip_path = os.path.join(venv_path, "bin", "pip")
                python_path = os.path.join(venv_path, "bin", "python")

            # Upgrade pip
            print("⬆️  Upgrading pip...")
            subprocess.run([pip_path, "install", "--upgrade", "pip"],
                         check=True, capture_output=True)

            # Install requirements with timing
            print(f"📥 Installing requirements from {filename}...")
            start_time = time.time()

            result = subprocess.run(
                [pip_path, "install", "-r", filename],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            install_time = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ Requirements installed successfully!")
                print(f"⏱️  Installation took {install_time:.2f} seconds")

                # Get list of installed packages
                list_result = subprocess.run([pip_path, "list", "--format=freeze"],
                                           capture_output=True, text=True)
                if list_result.returncode == 0:
                    installed_packages = list_result.stdout.strip().split('\n')
                    print(f"📊 Successfully installed {len(installed_packages)} packages")

                    # Show a few key packages
                    print("📦 Sample of installed packages:")
                    for pkg in installed_packages[:5]:
                        print(f"   • {pkg}")
                    if len(installed_packages) > 5:
                        print(f"   ... and {len(installed_packages) - 5} more packages")

                return True
            else:
                print("❌ Requirements installation failed:")
                print("STDERR:", result.stderr)
                if result.stdout:
                    print("STDOUT:", result.stdout)
                return False

        except subprocess.TimeoutExpired:
            print("❌ Installation timed out after 5 minutes")
            return False
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during testing: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print("Error details:", e.stderr)
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False


def check_pip_tools_compatibility(filename="requirements.txt"):
    """Optional: Check if pip-tools can resolve dependencies."""
    print("🔧 Checking pip-tools compatibility (optional)...")

    try:
        # Check if pip-tools is available
        result = subprocess.run([sys.executable, "-m", "pip", "show", "pip-tools"],
                              capture_output=True)

        if result.returncode != 0:
            print("ℹ️  pip-tools not installed, skipping dependency resolution check")
            return True

        # Try to compile requirements (dry run)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_output = temp_file.name

        try:
            result = subprocess.run([
                sys.executable, "-m", "piptools", "compile",
                "--dry-run", "--quiet", filename, "--output-file", temp_output
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print("✅ pip-tools dependency resolution successful")
                return True
            else:
                print("⚠️  pip-tools found potential dependency conflicts:")
                print(result.stderr)
                return False
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)

    except subprocess.TimeoutExpired:
        print("⚠️  pip-tools check timed out")
        return False
    except Exception as e:
        print(f"ℹ️  Could not run pip-tools check: {e}")
        return True  # Don't fail the whole test for this


def main():
    """Main test function."""
    print("🚀 Requirements.txt Compatibility Test")
    print("=" * 60)

    # Find requirements file
    requirements_files = ["requirements.txt", "requirements-dev.txt", "requirements-test.txt"]
    requirements_file = None

    for req_file in requirements_files:
        if os.path.exists(req_file):
            requirements_file = req_file
            break

    if not requirements_file:
        print("❌ No requirements file found!")
        print(f"   Looked for: {', '.join(requirements_files)}")
        sys.exit(1)

    print(f"📁 Using requirements file: {requirements_file}")
    print()

    # Step 1: Validate file exists and format
    if not validate_requirements_file(requirements_file):
        print("\n❌ File validation failed!")
        sys.exit(1)

    print()

    # Step 2: Check for version pattern issues
    version_warnings = check_version_patterns(requirements_file)

    print()

    # Step 3: Test actual installation
    install_success = test_requirements(requirements_file)

    print()

    # Step 4: Optional pip-tools check
    pip_tools_success = check_pip_tools_compatibility(requirements_file)

    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)

    if install_success:
        print("✅ Requirements.txt installation test PASSED!")

        if version_warnings:
            print(f"⚠️  {len(version_warnings)} version warnings were found")
            print("   These are usually not critical but worth reviewing")

        if pip_tools_success:
            print("✅ Dependency resolution check passed")
        else:
            print("⚠️  Dependency resolution had warnings")

        print("\n🎉 Your requirements.txt file should work correctly!")

    else:
        print("❌ Requirements.txt installation test FAILED!")
        print("\n💡 Suggestions:")
        print("   • Check for typos in package names")
        print("   • Verify version constraints are valid")
        print("   • Try installing packages individually to identify issues")
        print("   • Consider using 'pip freeze' to generate a new requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
