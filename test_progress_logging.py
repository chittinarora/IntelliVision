#!/usr/bin/env python3
"""
Test script for progress logging functionality.
This script demonstrates the standardized logging format for video analytics jobs.
"""

import time
import logging
import sys
import os

# Add the intellivision directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'intellivision'))

# Configure logging to show timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def test_video_progress_logging():
    """Test video processing progress logging."""
    print("🎬 Testing Video Processing Progress Logging")
    print("=" * 60)

    try:
        from apps.video_analytics.progress_logger import create_progress_logger

        # Simulate pothole detection with 1000 frames
        job_id = "123"
        total_frames = 1000
        progress_logger = create_progress_logger(
            job_id=job_id,
            total_items=total_frames,
            job_type="pothole_detection"
        )

        # Simulate processing frames
        for frame in range(0, total_frames + 1, 100):
            progress_logger.update_progress(
                frame,
                status="Processing video frames..."
            )
            time.sleep(0.1)  # Simulate processing time

        progress_logger.log_completion(total_frames)

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_image_progress_logging():
    """Test image processing progress logging."""
    print("\n🖼️ Testing Image Processing Progress Logging")
    print("=" * 60)

    try:
        from apps.video_analytics.progress_logger import create_progress_logger

        # Simulate food waste estimation with 4 images
        job_id = "abc123"
        total_images = 4
        progress_logger = create_progress_logger(
            job_id=job_id,
            total_items=total_images,
            job_type="food_waste_estimation"
        )

        # Simulate processing images
        for image in range(1, total_images + 1):
            progress_logger.update_progress(
                image,
                status="Awaiting API response..."
            )
            time.sleep(0.5)  # Simulate API call time

        progress_logger.log_completion(total_images)

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_batch_progress_logging():
    """Test batch processing progress logging."""
    print("\n📦 Testing Batch Processing Progress Logging")
    print("=" * 60)

    try:
        from apps.video_analytics.progress_logger import create_batch_progress_logger

        # Simulate batch processing
        job_id = "batch456"
        total_batches = 10
        items_per_batch = 50
        batch_logger = create_batch_progress_logger(
            job_id=job_id,
            total_batches=total_batches,
            items_per_batch=items_per_batch,
            job_type="people_count"
        )

        # Simulate batch processing
        for batch in range(1, total_batches + 1):
            batch_logger.update_batch_progress(
                batch,
                status="Processing batch..."
            )
            time.sleep(0.2)  # Simulate batch processing time

        batch_logger.log_completion()

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_error_logging():
    """Test error logging functionality."""
    print("\n❌ Testing Error Logging")
    print("=" * 60)

    try:
        from apps.video_analytics.progress_logger import create_progress_logger

        job_id = "error789"
        progress_logger = create_progress_logger(
            job_id=job_id,
            total_items=100,
            job_type="test_job"
        )

        # Simulate an error
        progress_logger.update_progress(50, status="Processing...")
        progress_logger.log_error("API connection failed")
        progress_logger.log_warning("Low confidence detection")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    """Run all progress logging tests."""
    print("🧪 Progress Logging Test Suite")
    print("=" * 60)
    print("This test demonstrates the standardized logging format for video analytics jobs.")
    print("Expected output format:")
    print("2025-07-26 02:38:00,123 INFO pothole_detection - **Job 123**: Progress **10.0%** (100/1000), Status: Processing video...")
    print("2025-07-26 02:38:00,124 INFO pothole_detection - [####------] Done: 00:10 | Left: 01:31 | Avg FPS: 9.8")
    print()

    # Run tests
    test_video_progress_logging()
    test_image_progress_logging()
    test_batch_progress_logging()
    test_error_logging()

    print("\n✅ All tests completed!")
    print("📝 Check the log output above to verify the standardized format.")

if __name__ == "__main__":
    main()
