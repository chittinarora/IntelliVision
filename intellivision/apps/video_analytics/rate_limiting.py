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
from typing import Dict, Deque, Tuple
from django.core.cache import cache
from django.conf import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter for API endpoints."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, Deque[float]] = defaultdict(lambda: deque())
        self.lock = Lock()

    def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        Check if request is allowed for the given key.

        Args:
            key: Unique identifier (e.g., user ID, IP address)

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        now = time.time()

        with self.lock:
            # Clean old requests
            while self.requests[key] and self.requests[key][0] < now - self.window_seconds:
                self.requests[key].popleft()

            # Check if limit exceeded
            if len(self.requests[key]) >= self.max_requests:
                return False, 0

            # Add current request
            self.requests[key].append(now)
            return True, self.max_requests - len(self.requests[key])


class ResourceManager:
    """Manages system resources to prevent exhaustion."""

    def __init__(self):
        self.active_jobs = 0
        self.max_concurrent_jobs = getattr(settings, 'MAX_CONCURRENT_JOBS', 10)
        self.lock = Lock()

    def can_start_job(self) -> bool:
        """Check if a new job can be started."""
        with self.lock:
            return self.active_jobs < self.max_concurrent_jobs

    def start_job(self) -> bool:
        """Start a new job if possible."""
        with self.lock:
            if self.active_jobs < self.max_concurrent_jobs:
                self.active_jobs += 1
                logger.info(f"Started job. Active jobs: {self.active_jobs}/{self.max_concurrent_jobs}")
                return True
            return False

    def end_job(self):
        """End a job and free up resources."""
        with self.lock:
            if self.active_jobs > 0:
                self.active_jobs -= 1
                logger.info(f"Ended job. Active jobs: {self.active_jobs}/{self.max_concurrent_jobs}")


# Global instances
rate_limiter = RateLimiter(max_requests=20, window_seconds=60)  # 20 requests per minute
resource_manager = ResourceManager()


def check_rate_limit(user_id: str) -> Tuple[bool, int]:
    """
    Check if user has exceeded rate limit.

    Args:
        user_id: User identifier

    Returns:
        Tuple of (is_allowed, remaining_requests)
    """
    return rate_limiter.is_allowed(f"user_{user_id}")


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


def get_system_status() -> Dict:
    """
    Get current system status for monitoring.

    Returns:
        Dictionary with system status information
    """
    return {
        'active_jobs': resource_manager.active_jobs,
        'max_concurrent_jobs': resource_manager.max_concurrent_jobs,
        'available_slots': resource_manager.max_concurrent_jobs - resource_manager.active_jobs,
        'utilization_percent': (resource_manager.active_jobs / resource_manager.max_concurrent_jobs) * 100
    }
