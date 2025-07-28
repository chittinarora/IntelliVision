"""
Management command to clean up old video jobs and associated files.
Helps prevent database and disk space from growing infinitely.
"""

import os
import logging
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.db import transaction
from apps.video_analytics.models import VideoJob

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Clean up old video jobs and associated files'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Delete jobs older than this many days (default: 30)'
        )
        parser.add_argument(
            '--status',
            type=str,
            choices=['completed', 'failed', 'all'],
            default='all',
            help='Only delete jobs with this status (default: all)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Process jobs in batches of this size (default: 100)'
        )

    def handle(self, *args, **options):
        days = options['days']
        status_filter = options['status']
        dry_run = options['dry_run']
        batch_size = options['batch_size']

        # Calculate cutoff date
        cutoff_date = timezone.now() - timedelta(days=days)

        self.stdout.write(
            self.style.SUCCESS(
                f"🧹 Starting cleanup of jobs older than {days} days "
                f"(before {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')})"
            )
        )

        if dry_run:
            self.stdout.write(
                self.style.WARNING("🔍 DRY RUN MODE - No actual deletions will occur")
            )

        # Build queryset
        queryset = VideoJob.objects.filter(created_at__lt=cutoff_date)

        if status_filter != 'all':
            queryset = queryset.filter(status=status_filter)

        total_jobs = queryset.count()

        if total_jobs == 0:
            self.stdout.write(
                self.style.SUCCESS("✅ No jobs found matching cleanup criteria")
            )
            return

        self.stdout.write(
            f"📊 Found {total_jobs} jobs matching cleanup criteria"
        )

        # Process jobs in batches
        processed = 0
        deleted_files = 0
        freed_space = 0

        while processed < total_jobs:
            batch = list(queryset[processed:processed + batch_size])

            if not batch:
                break

            with transaction.atomic():
                for job in batch:
                    files_to_delete = []
                    job_freed_space = 0

                    # Collect file paths
                    if job.input_video:
                        try:
                            file_path = job.input_video.path
                            if os.path.exists(file_path):
                                files_to_delete.append(file_path)
                                job_freed_space += os.path.getsize(file_path)
                        except (ValueError, AttributeError):
                            pass

                    if job.output_video:
                        try:
                            file_path = job.output_video.path
                            if os.path.exists(file_path):
                                files_to_delete.append(file_path)
                                job_freed_space += os.path.getsize(file_path)
                        except (ValueError, AttributeError):
                            pass

                    if job.output_image:
                        try:
                            file_path = job.output_image.path
                            if os.path.exists(file_path):
                                files_to_delete.append(file_path)
                                job_freed_space += os.path.getsize(file_path)
                        except (ValueError, AttributeError):
                            pass

                    # Show what would be deleted
                    if dry_run:
                        self.stdout.write(
                            f"  📋 Job {job.id} ({job.status}) - "
                            f"{len(files_to_delete)} files, "
                            f"{self._format_size(job_freed_space)}"
                        )
                        for file_path in files_to_delete:
                            self.stdout.write(f"    📄 {file_path}")
                    else:
                        # Actually delete files
                        for file_path in files_to_delete:
                            try:
                                os.remove(file_path)
                                deleted_files += 1
                                logger.info(f"Deleted file: {file_path}")
                            except OSError as e:
                                logger.warning(f"Failed to delete file {file_path}: {e}")

                        # Delete job record
                        job.delete()
                        logger.info(f"Deleted job {job.id}")

                    freed_space += job_freed_space
                    processed += 1

                    # Progress indicator
                    if processed % 10 == 0:
                        progress = (processed / total_jobs) * 100
                        self.stdout.write(
                            f"  ⏳ Progress: {processed}/{total_jobs} ({progress:.1f}%)"
                        )

        # Summary
        if dry_run:
            self.stdout.write(
                self.style.SUCCESS(
                    f"🔍 DRY RUN SUMMARY:\n"
                    f"  📊 Would delete {total_jobs} jobs\n"
                    f"  📄 Would delete {deleted_files} files\n"
                    f"  💾 Would free {self._format_size(freed_space)}\n"
                    f"  ⚡ Run without --dry-run to perform actual cleanup"
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f"✅ CLEANUP COMPLETED:\n"
                    f"  📊 Deleted {processed} jobs\n"
                    f"  📄 Deleted {deleted_files} files\n"
                    f"  💾 Freed {self._format_size(freed_space)}"
                )
            )

        # Recommendations
        if total_jobs > 1000:
            self.stdout.write(
                self.style.WARNING(
                    "⚠️  Large number of jobs found. Consider:\n"
                    "  • Running cleanup more frequently\n"
                    "  • Adjusting the retention period\n"
                    "  • Setting up automated cleanup via cron"
                )
            )

    def _format_size(self, size_bytes):
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"

        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0

        return f"{size_bytes:.1f} PB"
