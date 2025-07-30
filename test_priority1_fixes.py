#!/usr/bin/env python3
"""
Test script to verify Priority 1 fixes for parameter count mismatches.
Tests emergency_count.py and lobby_detection.py function signatures.
"""

import os
import sys
import inspect
import tempfile

# Add the project root to the Python path
project_root = '/Users/adidubbs/Desktop/IntelliVision'
sys.path.insert(0, project_root)

# Import the functions we want to test
try:
    from intellivision.apps.video_analytics.analytics.emergency_count import (
        run_optimal_yolov12x_counting,
        tracking_video as emergency_tracking_video
    )
    from intellivision.apps.video_analytics.analytics.lobby_detection import (
        run_crowd_analysis,
        tracking_video as lobby_tracking_video
    )
    print("✅ Successfully imported all functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_function_signatures():
    """Test that function signatures accept the expected number of parameters."""
    print("\n🔍 Testing Function Signatures...")

    # Test emergency_count.py functions
    print("\n1. Emergency Count Functions:")

    # Test run_optimal_yolov12x_counting signature
    sig = inspect.signature(run_optimal_yolov12x_counting)
    params = list(sig.parameters.keys())
    print(f"   run_optimal_yolov12x_counting parameters: {params}")

    expected_params = ['video_path', 'line_definitions', 'custom_params', 'output_path', 'job_id']
    if params == expected_params:
        print("   ✅ run_optimal_yolov12x_counting signature correct")
    else:
        print(f"   ❌ run_optimal_yolov12x_counting signature incorrect. Expected: {expected_params}")

    # Test emergency tracking_video signature
    sig = inspect.signature(emergency_tracking_video)
    params = list(sig.parameters.keys())
    print(f"   emergency tracking_video parameters: {params}")

    expected_params = ['video_path', 'output_path', 'line_configs', 'video_width', 'video_height', 'job_id']
    if params == expected_params:
        print("   ✅ emergency tracking_video signature correct")
    else:
        print(f"   ❌ emergency tracking_video signature incorrect. Expected: {expected_params}")

    # Test lobby_detection.py functions
    print("\n2. Lobby Detection Functions:")

    # Test run_crowd_analysis signature
    sig = inspect.signature(run_crowd_analysis)
    params = list(sig.parameters.keys())
    print(f"   run_crowd_analysis parameters: {params}")

    expected_params = ['source_path', 'zone_configs', 'output_path', 'job_id']
    if params == expected_params:
        print("   ✅ run_crowd_analysis signature correct")
    else:
        print(f"   ❌ run_crowd_analysis signature incorrect. Expected: {expected_params}")

    # Test lobby tracking_video signature
    sig = inspect.signature(lobby_tracking_video)
    params = list(sig.parameters.keys())
    print(f"   lobby tracking_video parameters: {params}")

    expected_params = ['source_path', 'zone_configs', 'output_path', 'job_id']
    if params == expected_params:
        print("   ✅ lobby tracking_video signature correct")
    else:
        print(f"   ❌ lobby tracking_video signature incorrect. Expected: {expected_params}")

def test_function_calls():
    """Test that functions can be called with the correct number of arguments."""
    print("\n🧪 Testing Function Calls...")

    # Mock parameters for testing
    video_path = "test_video.mp4"
    output_path = "/tmp/test_output.mp4"
    job_id = "test_job_123"

    # Emergency count test parameters
    line_definitions = {
        "line1": {
            "coords": [[100, 200], [300, 200]],
            "inDirection": "UP"
        }
    }

    # Lobby detection test parameters
    zone_configs = {
        "zone1": {
            "points": [[0, 0], [100, 0], [100, 100], [0, 100]]
        }
    }

    print("\n1. Testing parameter count compatibility:")

    # Test emergency count function calls
    try:
        # This should not raise TypeError for wrong number of arguments
        sig = inspect.signature(run_optimal_yolov12x_counting)
        sig.bind(video_path, line_definitions, None, output_path, job_id)
        print("   ✅ run_optimal_yolov12x_counting accepts 5 arguments")
    except TypeError as e:
        print(f"   ❌ run_optimal_yolov12x_counting parameter error: {e}")

    try:
        sig = inspect.signature(emergency_tracking_video)
        sig.bind(video_path, output_path, line_definitions, 1920, 1080, job_id)
        print("   ✅ emergency tracking_video accepts 6 arguments")
    except TypeError as e:
        print(f"   ❌ emergency tracking_video parameter error: {e}")

    # Test lobby detection function calls
    try:
        sig = inspect.signature(run_crowd_analysis)
        sig.bind(video_path, zone_configs, output_path, job_id)
        print("   ✅ run_crowd_analysis accepts 4 arguments")
    except TypeError as e:
        print(f"   ❌ run_crowd_analysis parameter error: {e}")

    try:
        sig = inspect.signature(lobby_tracking_video)
        sig.bind(video_path, zone_configs, output_path, job_id)
        print("   ✅ lobby tracking_video accepts 4 arguments")
    except TypeError as e:
        print(f"   ❌ lobby tracking_video parameter error: {e}")

def test_tasks_integration():
    """Test that the functions match what tasks.py expects to call."""
    print("\n🔗 Testing Tasks.py Integration...")

    # Simulate what tasks.py does
    job_id = "123"

    # Emergency count: tasks.py passes [video_path, output_path, emergency_lines, video_width, video_height] + [job_id]
    emergency_args = ["test_video.mp4", "/tmp/output.mp4", {"line1": {"coords": [[0,0], [100,100]]}}, 1920, 1080, job_id]

    try:
        sig = inspect.signature(emergency_tracking_video)
        sig.bind(*emergency_args)
        print("   ✅ Emergency count matches tasks.py call pattern")
    except TypeError as e:
        print(f"   ❌ Emergency count tasks.py integration error: {e}")

    # Lobby detection: tasks.py passes [video_path, zone_configs, output_path] + [job_id]
    lobby_args = ["test_video.mp4", {"zone1": {"points": [[0,0], [100,100]]}}, "/tmp/output.mp4", job_id]

    try:
        sig = inspect.signature(lobby_tracking_video)
        sig.bind(*lobby_args)
        print("   ✅ Lobby detection matches tasks.py call pattern")
    except TypeError as e:
        print(f"   ❌ Lobby detection tasks.py integration error: {e}")

def main():
    """Run all tests."""
    print("🚀 Testing Priority 1 Fixes for Parameter Count Mismatches")
    print("=" * 60)

    test_function_signatures()
    test_function_calls()
    test_tasks_integration()

    print("\n" + "=" * 60)
    print("✅ Priority 1 testing completed!")
    print("\nIf all tests show ✅, the parameter count mismatches have been resolved.")

if __name__ == "__main__":
    main()
