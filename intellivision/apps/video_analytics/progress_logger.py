# /apps/video_analytics/progress_logger.py

"""
=====================================
Progress Logger for Video Analytics Jobs
=====================================
Provides standardized progress tracking with detailed timing, progress bars, and status updates.
"""

import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from .progress_utils import update_job_progress


class ProgressLogger:
    """Standardized progress logger for video analytics jobs."""

    def __init__(self, job_id: str, total_items: int, job_type: str, logger_name: str = None):
        """
        Initialize progress logger.

        Args:
            job_id: Unique job identifier
            total_items: Total number of items to process (frames, images, etc.)
            job_type: Type of analytics job
            logger_name: Custom logger name (defaults to job_type)
        """
        self.job_id = job_id
        self.total_items = total_items
        self.job_type = job_type
        self.logger = logging.getLogger(logger_name or job_type)

        # Timing and progress tracking
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.processed_items = 0
        self.last_processed_count = 0

        # Performance tracking
        self.processing_times = []
        self.status = "Initializing..."

        # Log initial setup
        self.logger.info(f"**Job {job_id}**: Starting {job_type} processing ({total_items} items)")

    def update_progress(self, processed_count: int, status: str = None, force_log: bool = False):
        """
        Update progress and log if significant progress made.

        Args:
            processed_count: Number of items processed so far
            status: Current status message
            force_log: Force logging even if minimal progress
        """
        current_time = time.time()
        self.processed_items = processed_count

        if status:
            self.status = status

        # Calculate progress percentage
        progress_percent = (processed_count / self.total_items) * 100 if self.total_items > 0 else 0

        # Determine if we should log (every 10% or every 100 items for large jobs)
        should_log = force_log
        if not should_log:
            if self.total_items <= 100:
                # For small jobs, log every item
                should_log = processed_count > self.last_processed_count
            elif self.total_items <= 1000:
                # For medium jobs, log every 10% or 50 items
                should_log = (progress_percent - (self.last_processed_count / self.total_items) * 100 >= 10) or \
                           (processed_count - self.last_processed_count >= 50)
            else:
                # For large jobs, log every 5% or 100 items
                should_log = (progress_percent - (self.last_processed_count / self.total_items) * 100 >= 5) or \
                           (processed_count - self.last_processed_count >= 100)

        if should_log:
            self._log_progress(processed_count, progress_percent, current_time)

            # Update progress in database for real-time frontend tracking
            try:
                update_job_progress(
                    job_id=int(self.job_id),
                    processed_frames=processed_count,
                    total_frames=self.total_items,
                    fps=avg_rate if avg_rate > 0 else None
                )
            except Exception as e:
                self.logger.warning(f"Failed to update database progress: {e}")

            self.last_processed_count = processed_count
            self.last_update_time = current_time

    def _log_progress(self, processed_count: int, progress_percent: float, current_time: float):
        """Log detailed progress information."""
        # Calculate timing information
        elapsed_time = current_time - self.start_time
        if processed_count > 0:
            avg_rate = processed_count / elapsed_time
            remaining_items = self.total_items - processed_count
            estimated_remaining = remaining_items / avg_rate if avg_rate > 0 else 0
        else:
            avg_rate = 0
            estimated_remaining = 0

        # Format timing strings
        elapsed_str = self._format_time(elapsed_time)
        remaining_str = self._format_time(estimated_remaining)

        # Create progress bar
        progress_bar = self._create_progress_bar(progress_percent)

        # Determine rate unit based on job type
        if "image" in self.job_type.lower() or "food" in self.job_type.lower():
            rate_unit = "Images/s"
        else:
            rate_unit = "FPS"

        # Log the progress
        self.logger.info(f"**Job {self.job_id}**: Progress **{progress_percent:.1f}%** ({processed_count}/{self.total_items}), Status: {self.status}")
        self.logger.info(f"{progress_bar} Done: {elapsed_str} | Left: {remaining_str} | Avg {rate_unit}: {avg_rate:.1f}")

    def _create_progress_bar(self, progress_percent: float, width: int = 10) -> str:
        """Create a visual progress bar."""
        filled = int((progress_percent / 100) * width)
        empty = width - filled
        return f"[{'#' * filled}{'-' * empty}]"

    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS format."""
        if seconds < 0:
            return "00:00"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def log_completion(self, final_count: int = None):
        """Log job completion."""
        if final_count is None:
            final_count = self.processed_items

        total_time = time.time() - self.start_time
        total_time_str = self._format_time(total_time)

        if final_count > 0:
            avg_rate = final_count / total_time
            if "image" in self.job_type.lower() or "food" in self.job_type.lower():
                rate_unit = "Images/s"
            else:
                rate_unit = "FPS"
        else:
            avg_rate = 0
            rate_unit = "FPS"

        self.logger.info(f"**Job {self.job_id}**: Completed {self.job_type} processing")
        self.logger.info(f"Total time: {total_time_str} | Final {rate_unit}: {avg_rate:.1f} | Items processed: {final_count}")

    def log_error(self, error_message: str):
        """Log error with job context."""
        self.logger.error(f"**Job {self.job_id}**: Error in {self.job_type} - {error_message}")

    def log_warning(self, warning_message: str):
        """Log warning with job context."""
        self.logger.warning(f"**Job {self.job_id}**: Warning in {self.job_type} - {warning_message}")


class BatchProgressLogger:
    """Progress logger for batch processing operations."""

    def __init__(self, job_id: str, total_batches: int, items_per_batch: int, job_type: str):
        """
        Initialize batch progress logger.

        Args:
            job_id: Unique job identifier
            total_batches: Total number of batches
            items_per_batch: Items per batch
            job_type: Type of analytics job
        """
        self.job_id = job_id
        self.total_batches = total_batches
        self.items_per_batch = items_per_batch
        self.total_items = total_batches * items_per_batch
        self.job_type = job_type
        self.logger = logging.getLogger(job_type)

        self.start_time = time.time()
        self.completed_batches = 0
        self.status = "Initializing batch processing..."

        self.logger.info(f"**Job {job_id}**: Starting {job_type} batch processing ({total_batches} batches, {self.total_items} total items)")

    def update_batch_progress(self, completed_batches: int, status: str = None):
        """Update batch progress."""
        self.completed_batches = completed_batches
        if status:
            self.status = status

        progress_percent = (completed_batches / self.total_batches) * 100 if self.total_batches > 0 else 0
        processed_items = completed_batches * self.items_per_batch

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if completed_batches > 0:
            avg_rate = completed_batches / elapsed_time
            remaining_batches = self.total_batches - completed_batches
            estimated_remaining = remaining_batches / avg_rate if avg_rate > 0 else 0
        else:
            avg_rate = 0
            estimated_remaining = 0

        elapsed_str = self._format_time(elapsed_time)
        remaining_str = self._format_time(estimated_remaining)
        progress_bar = self._create_progress_bar(progress_percent)

        self.logger.info(f"**Job {self.job_id}**: Progress **{progress_percent:.1f}%** ({completed_batches}/{self.total_batches} batches), Status: {self.status}")
        self.logger.info(f"{progress_bar} Done: {elapsed_str} | Left: {remaining_str} | Avg Batches/s: {avg_rate:.2f}")

    def _create_progress_bar(self, progress_percent: float, width: int = 10) -> str:
        """Create a visual progress bar."""
        filled = int((progress_percent / 100) * width)
        empty = width - filled
        return f"[{'#' * filled}{'-' * empty}]"

    def _format_time(self, seconds: float) -> str:
        """Format time in MM:SS format."""
        if seconds < 0:
            return "00:00"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def log_completion(self):
        """Log batch completion."""
        total_time = time.time() - self.start_time
        total_time_str = self._format_time(total_time)

        if self.completed_batches > 0:
            avg_rate = self.completed_batches / total_time
        else:
            avg_rate = 0

        self.logger.info(f"**Job {self.job_id}**: Completed {self.job_type} batch processing")
        self.logger.info(f"Total time: {total_time_str} | Avg Batches/s: {avg_rate:.2f} | Batches completed: {self.completed_batches}")


def create_progress_logger(job_id: str, total_items: int, job_type: str, logger_name: str = None) -> ProgressLogger:
    """
    Factory function to create a progress logger.

    Args:
        job_id: Unique job identifier
        total_items: Total number of items to process
        job_type: Type of analytics job
        logger_name: Custom logger name

    Returns:
        ProgressLogger instance
    """
    return ProgressLogger(job_id, total_items, job_type, logger_name)


def create_batch_progress_logger(job_id: str, total_batches: int, items_per_batch: int, job_type: str) -> BatchProgressLogger:
    """
    Factory function to create a batch progress logger.

    Args:
        job_id: Unique job identifier
        total_batches: Total number of batches
        items_per_batch: Items per batch
        job_type: Type of analytics job

    Returns:
        BatchProgressLogger instance
    """
    return BatchProgressLogger(job_id, total_batches, items_per_batch, job_type)
