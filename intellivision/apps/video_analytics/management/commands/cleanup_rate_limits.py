"""
Management command to clean up stale rate limits and job slots.
Run this periodically to prevent resource leaks.
"""

from django.core.management.base import BaseCommand
from django.core.cache import cache
from apps.video_analytics.rate_limiting import cleanup_stale_jobs, get_system_status
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Clean up stale rate limits and job slots'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force cleanup even if not needed',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be cleaned without actually doing it',
        )

    def handle(self, *args, **options):
        self.stdout.write("🧹 Starting rate limit cleanup...")

        try:
            # Get current system status
            status = get_system_status()

            if options['dry_run']:
                self.stdout.write(f"📊 Current status: {status}")
                self.stdout.write("🔍 Dry run - no changes made")
                return

            # Clean up stale jobs
            cleanup_stale_jobs()

            # Get updated status
            new_status = get_system_status()

            self.stdout.write(
                self.style.SUCCESS(
                    f"✅ Cleanup completed. Active jobs: {status.get('active_jobs', 0)} -> {new_status.get('active_jobs', 0)}"
                )
            )

            if options['verbose']:
                self.stdout.write(f"📊 System status: {new_status}")

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Cleanup failed: {e}")
            )
            logger.error(f"Rate limit cleanup failed: {e}")
