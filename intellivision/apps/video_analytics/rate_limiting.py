# /apps/video_analytics/rate_limiting.py

"""
=====================================
Rate Limiting Utilities
=====================================
Provides rate limiting functionality to prevent resource exhaustion and DoS attacks.
"""

import time
import logging
from collections import defaultdict, deque
from threading import Lock
from typing import Dict, Deque, Tuple, Any
from django.core.cache import cache
from django.conf import settings

logger = logging.getLogger(__name__)





class ResourceManager:
    """Manages system resources to prevent exhaustion using Redis."""

    def __init__(self):
        # VM Specs: 6 vCPUs, 27GB RAM, Tesla P100 16GB
        self.max_concurrent_jobs = getattr(settings, 'MAX_CONCURRENT_JOBS', 6)  # Match 6 vCPUs
        self.max_gpu_memory_gb = getattr(settings, 'MAX_GPU_MEMORY_GB', 14.0)  # Tesla P100 has 16GB, use 14GB (87.5%)
        self.max_system_memory_gb = getattr(settings, 'MAX_SYSTEM_MEMORY_GB', 24.0)  # 27GB total, reserve 3GB for system

    def can_start_job(self) -> bool:
        """Check if a new job can be started."""
        try:
            # Check CPU/memory resources using Redis
            active_jobs = cache.get('active_jobs', 0)
            if active_jobs >= self.max_concurrent_jobs:
                return False

            # Check GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    if gpu_memory_gb > self.max_gpu_memory_gb:
                        logger.warning(f"GPU memory limit exceeded: {gpu_memory_gb:.2f}GB > {self.max_gpu_memory_gb}GB")
                        return False
            except Exception as e:
                logger.warning(f"GPU memory check failed: {e}")

            # Check system memory
            try:
                import psutil
                system_memory_gb = psutil.virtual_memory().used / (1024 ** 3)
                if system_memory_gb > self.max_system_memory_gb:
                    logger.warning(f"System memory limit exceeded: {system_memory_gb:.2f}GB > {self.max_system_memory_gb}GB")
                    return False
            except Exception as e:
                logger.warning(f"System memory check failed: {e}")

            return True
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return True  # Allow if Redis is unavailable

    def start_job(self) -> bool:
        """Start a new job if possible."""
        try:
            active_jobs = cache.get('active_jobs', 0)
            if active_jobs < self.max_concurrent_jobs:
                cache.set('active_jobs', active_jobs + 1, timeout=3600)  # 1 hour timeout
                logger.info(f"Started job. Active jobs: {active_jobs + 1}/{self.max_concurrent_jobs}")
                return True
            return False
        except Exception as e:
            logger.error(f"Job start failed: {e}")
            return True  # Allow if Redis is unavailable

    def end_job(self):
        """End a job and free up resources."""
        try:
            active_jobs = cache.get('active_jobs', 0)
            if active_jobs > 0:
                cache.set('active_jobs', active_jobs - 1, timeout=3600)
                logger.info(f"Ended job. Active jobs: {active_jobs - 1}/{self.max_concurrent_jobs}")
        except Exception as e:
            logger.error(f"Job end failed: {e}")


# Resource manager for system resources
resource_manager = ResourceManager()


def check_resource_availability() -> bool:
    """
    Check if system has resources available for new jobs.

    Returns:
        True if resources are available
    """
    return resource_manager.can_start_job()


def acquire_job_slot() -> bool:
    """
    Try to acquire a job slot.

    Returns:
        True if slot was acquired
    """
    return resource_manager.start_job()


def release_job_slot():
    """Release a job slot."""
    resource_manager.end_job()


def cleanup_stale_jobs():
    """
    Simple cleanup of stale job slots.
    """
    try:
        active_jobs = cache.get('active_jobs', 0)
        if active_jobs > resource_manager.max_concurrent_jobs * 2:
            logger.warning(f"Detected {active_jobs} active jobs, resetting to {resource_manager.max_concurrent_jobs}")
            cache.set('active_jobs', resource_manager.max_concurrent_jobs, timeout=3600)
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")




def get_system_status() -> Dict:
    """
    Get current system status for monitoring.

    Returns:
        Dictionary with system status information
    """
    try:
        active_jobs = cache.get('active_jobs', 0)
        status = {
            'active_jobs': active_jobs,
            'max_concurrent_jobs': resource_manager.max_concurrent_jobs,
            'available_slots': resource_manager.max_concurrent_jobs - active_jobs,
            'utilization_percent': (active_jobs / resource_manager.max_concurrent_jobs) * 100 if resource_manager.max_concurrent_jobs > 0 else 0
        }

        # Add GPU memory info if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                status.update({
                    'gpu_memory_used_gb': round(gpu_memory_gb, 2),
                    'gpu_memory_total_gb': round(gpu_memory_total_gb, 2),
                    'gpu_memory_percent': round((gpu_memory_gb / gpu_memory_total_gb) * 100, 1),
                    'gpu_memory_limit_gb': resource_manager.max_gpu_memory_gb
                })
        except Exception as e:
            logger.warning(f"GPU status check failed: {e}")
            status['gpu_status'] = 'unavailable'

        return status
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return {
            'active_jobs': 0,
            'max_concurrent_jobs': resource_manager.max_concurrent_jobs,
            'available_slots': resource_manager.max_concurrent_jobs,
            'utilization_percent': 0,
            'error': 'Status unavailable'
        }
