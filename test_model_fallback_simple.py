#!/usr/bin/env python3
"""
Simple test script for model manager fallback logic without Django.
"""
import os
from pathlib import Path

# Mock Django settings for testing
class MockSettings:
    MEDIA_ROOT = str(Path(__file__).parent / "media")

# Mock the django.conf.settings
import sys
sys.modules['django'] = type(sys)('django')
sys.modules['django.conf'] = type(sys)('django.conf')
sys.modules['django.conf'].settings = MockSettings()

# Now import the model manager
sys.path.insert(0, str(Path(__file__).parent / "intellivision" / "apps"))

from video_analytics.analytics.model_manager import (
    MODEL_CONFIGS,
    get_model_path,
    get_model_with_fallback,
    resolve_model_config,
    MODELS_DIR
)

def test_model_configs():
    """Test the model configurations"""
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
        filename = config.get('filename', 'N/A')
        print(f"  • {name:<15} → file: {filename:<20} fallback: {fallback:<10} skip: {skip_download}")
    
    print("\n🔄 Testing get_model_path:")
    test_models = ["yolov8x", "yolo11m", "best_car", "best_animal", "nonexistent"]
    
    for model_name in test_models:
        path = get_model_path(model_name)
        exists = path.exists() if path else False
        print(f"  • {model_name:<15} → {path} (exists: {exists})")
    
    print("\n🔄 Testing Model Resolution with Fallback:")
    
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
            print(f"  ❌ Unexpected error: {type(e).__name__}: {e}")

def test_fallback_scenario():
    """Test specific fallback scenario"""
    print("\n" + "=" * 50)
    print("🎯 Testing Specific Fallback Scenario:")
    
    # Simulate missing best_car.pt
    best_car_path = get_model_path("best_car")
    print(f"best_car.pt path: {best_car_path}")
    print(f"best_car.pt exists: {best_car_path.exists() if best_car_path else 'N/A'}")
    
    # Check fallback model
    yolo11m_path = get_model_path("yolo11m")
    print(f"yolo11m.pt path: {yolo11m_path}")
    print(f"yolo11m.pt exists: {yolo11m_path.exists() if yolo11m_path else 'N/A'}")
    
    # Test what happens when best_car is missing
    print(f"\nWhat happens when best_car.pt is missing:")
    try:
        config = MODEL_CONFIGS["best_car"]
        print(f"best_car config: {config}")
        fallback_name = config.get("fallback")
        print(f"Fallback model: {fallback_name}")
        
        if fallback_name:
            fallback_path = get_model_path(fallback_name)
            print(f"Fallback path: {fallback_path}")
            print(f"Fallback exists: {fallback_path.exists() if fallback_path else 'N/A'}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_model_configs()
    test_fallback_scenario()