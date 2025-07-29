"""
=====================================
Progress Logger for Video Analytics Jobs
=====================================
Provides standardized progress tracking with both one-line progress bars
and detailed, sectioned milestone logs. Designed for clear, readable logs in both
real-time and for milestone events.
"""

import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from .progress_utils import update_job_progress

class ProgressLogger:
    """Standardized progress logger for video analytics jobs (supports pretty bar + section block)."""

    def __init__(self, job_id: str, total_items: int, job_type: str, logger_name: str = None):
        """
        Initialize progress logger.
        """
        self.job_id = job_id
        self.total_items = total_items
        self.job_type = job_type
        self.logger = logging.getLogger(logger_name or job_type)
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.processed_items = 0
        self.last_processed_count = 0
        self.status = "Initializing..."
        self._log_section("Job started", extra_status="Starting job...")

    def update_progress(self, processed_count: int, status: str = None, force_log: bool = False):
        """
        Update progress (frequent calls) and log pretty bar.
        """
        current_time = time.time()
        self.processed_items = processed_count
        if status:
            self.status = status
        progress_percent = (processed_count / self.total_items) * 100 if self.total_items > 0 else 0

        # Check for 25% milestone (including 100%)
        current_milestone = int(progress_percent // 25) * 25
        last_milestone = int((self.last_processed_count / self.total_items) * 100 // 25) * 25 if self.total_items > 0 else 0

        # Log sectioned milestone at every 25% (including 100%)
        if current_milestone > last_milestone and current_milestone > 0:
            milestone_msg = "Job completed ✔️" if current_milestone == 100 else f"Milestone: {current_milestone}% complete"
            self.log_section(milestone_msg)

        # Log pretty bar at regular intervals
        should_log = force_log
        if not should_log:
            if self.total_items <= 100:
                should_log = processed_count > self.last_processed_count
            elif self.total_items <= 1000:
                should_log = (progress_percent - (self.last_processed_count / self.total_items) * 100 >= 10) or \
                             (processed_count - self.last_processed_count >= 50)
            else:
                should_log = (progress_percent - (self.last_processed_count / self.total_items) * 100 >= 5) or \
                             (processed_count - self.last_processed_count >= 100)

        if should_log:
            self.log_pretty_bar()
            # Optionally update database/Redis for UI tracking
            try:
                avg_rate = self._calc_avg_rate()
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

    def log_pretty_bar(self):
        """Log a compact, pretty progress bar on one line."""
        elapsed = time.time() - self.start_time
        avg_rate = self._calc_avg_rate()
        remaining = self._estimate_remaining_time(avg_rate)
        progress_percent = (self.processed_items / self.total_items) * 100 if self.total_items else 0
        progress_bar = self._create_bar(progress_percent, width=12)
        status_emoji = self._get_status_emoji()
        self.logger.info(
            f"Job #{self.job_id} | {self.job_type} | "
            f"{progress_bar} {progress_percent:3.0f}% | "
            f"⏱️ {self._format_time(elapsed)} done, {self._format_time(remaining)} left | "
            f"{avg_rate:.1f} FPS | {status_emoji}"
        )

    def log_section(self, extra_status: str = None):
        """Log a full, detailed sectioned progress block (milestone/event)."""
        elapsed = time.time() - self.start_time
        avg_rate = self._calc_avg_rate()
        remaining = self._estimate_remaining_time(avg_rate)
        progress_percent = (self.processed_items / self.total_items) * 100 if self.total_items else 0

        separator = "=" * 13 + f" [JOB {self.job_id}: {self.job_type}] " + "=" * 13
        self.logger.info(separator)
        self.logger.info(f"Progress    : {progress_percent:>3.0f}%  ({self.processed_items} / {self.total_items})")
        self.logger.info(f"Elapsed     : {self._format_time(elapsed)}")
        self.logger.info(f"ETA         : {self._format_time(remaining)}")
        self.logger.info(f"Speed       : {avg_rate:.1f} FPS")
        self.logger.info(f"Status      : {extra_status or self.status}")
        self.logger.info("=" * len(separator))

    def log_completion(self, final_count: int = None):
        """Log job completion in both styles."""
        if final_count is not None:
            self.processed_items = final_count
        self.log_pretty_bar()
        self._log_section("Job completed ✔️")

    def log_error(self, error_message: str):
        """Log error with both bar and section block."""
        self.status = f"Error: {error_message}"
        self.log_pretty_bar()
        self._log_section(f"Error: {error_message}")

    def log_warning(self, warning_message: str):
        """Log warning as a sectioned block."""
        self._log_section(f"Warning: {warning_message}")

    # ----- Helper methods -----

    def _calc_avg_rate(self):
        elapsed = time.time() - self.start_time
        return self.processed_items / elapsed if elapsed > 0 else 0

    def _estimate_remaining_time(self, avg_rate):
        remaining = self.total_items - self.processed_items
        return remaining / avg_rate if avg_rate > 0 else 0

    def _create_bar(self, progress_percent: float, width: int = 10) -> str:
        filled = int((progress_percent / 100) * width)
        empty = width - filled
        return "[" + "█" * filled + "░" * empty + "]"

    def _format_time(self, seconds: float) -> str:
        if seconds is None or seconds < 0:
            return "--:--"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _get_status_emoji(self):
        if "error" in self.status.lower():
            return "🔴"
        elif "warning" in self.status.lower():
            return "🟡"
        elif "completed" in self.status.lower():
            return "✅"
        else:
            return "🟢"

    def _log_section(self, extra_status: str = None):
        """Helper: sectioned block for start/finish/error."""
        elapsed = time.time() - self.start_time
        avg_rate = self._calc_avg_rate()
        remaining = self._estimate_remaining_time(avg_rate)
        progress_percent = (self.processed_items / self.total_items) * 100 if self.total_items else 0
        separator = "=" * 13 + f" [JOB {self.job_id}: {self.job_type}] " + "=" * 13
        self.logger.info(separator)
        self.logger.info(f"Progress    : {progress_percent:>3.0f}%  ({self.processed_items} / {self.total_items})")
        self.logger.info(f"Elapsed     : {self._format_time(elapsed)}")
        self.logger.info(f"ETA         : {self._format_time(remaining)}")
        self.logger.info(f"Speed       : {avg_rate:.1f} FPS")
        self.logger.info(f"Status      : {extra_status or self.status}")
        self.logger.info("=" * len(separator))

# Factory functions remain
def create_progress_logger(job_id: str, total_items: int, job_type: str, logger_name: str = None) -> ProgressLogger:
    return ProgressLogger(job_id, total_items, job_type, logger_name)
