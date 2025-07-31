#!/usr/bin/env python3
"""
Test script for model manager fallback logic.
"""
import os
import sys
from pathlib import Path

# Add the IntelliVision directory to Python path
project_root = Path(__file__).parent / "intellivision"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "apps"))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'intellivision.settings')

import django
django.setup()

# Now we can import from the analytics modules
from intellivision.apps.video_analytics.analytics.model_manager import (
    get_model_with_fallback, 
    resolve_model_config,
    MODEL_CONFIGS,
    MODELS_DIR
)

def test_model_fallback():
    """Test the model fallback logic"""
    print("🧪 Testing Model Manager Fallback Logic")
    print("=" * 50)
    
    # Print current models directory
    print(f"📁 Models directory: {MODELS_DIR}")
    print(f"Directory exists: {MODELS_DIR.exists()}")
    
    if MODELS_DIR.exists():
        print(f"Contents: {list(MODELS_DIR.glob('*.pt'))}")
    
    print("\n📋 Available Model Configurations:")
    for name, config in MODEL_CONFIGS.items():
        fallback = config.get('fallback', 'None')
        skip_download = config.get('skip_download', False)
        print(f"  • {name} → fallback: {fallback}, skip_download: {skip_download}")
    
    print("\n🔄 Testing Model Resolution:")
    
    # Test cases
    test_cases = [
        ("yolov8x", "Should work - standard model"),
        ("yolo11m", "Should work or fallback to yolov11x"),
        ("best_car", "Should fallback to yolo11m (custom model)"),
        ("best_animal", "Should fallback to yolov11x (custom model)"),
        ("nonexistent", "Should fail - no such model"),
    ]
    
    for model_name, description in test_cases:
        print(f"\n🧪 Testing '{model_name}' - {description}")
        try:
            # Test with auto_download=False to avoid downloading
            model_path = get_model_with_fallback(model_name, auto_download=False)
            config = resolve_model_config(model_name, auto_download=False)
            
            print(f"  ✅ Success: {model_path}")
            print(f"  📊 Config: name={config['name']}, is_fallback={config['is_fallback']}")
            
        except FileNotFoundError as e:
            print(f"  ❌ FileNotFoundError: {e}")
        except ValueError as e:
            print(f"  ❌ ValueError: {e}")
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_model_fallback()