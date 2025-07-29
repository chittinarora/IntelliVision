#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VMTestRunner:
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        os.makedirs('test_reports', exist_ok=True)

    def run_command(self, command, description):
        logger.info(f"Running {description}...")
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Error in {description}: {e.stderr}")
            return False, e.stderr

    def check_apache_status(self):
        """Check if Apache is running and configured correctly"""
        return self.run_command(
            "systemctl status apache2",
            "Apache status check"
        )

    def check_apache_logs(self):
        """Check Apache error logs for issues"""
        return self.run_command(
            "tail -n 50 /var/log/apache2/error.log",
            "Apache error log check"
        )

    def check_wsgi_config(self):
        """Verify WSGI configuration"""
        return self.run_command(
            "apache2ctl -M | grep wsgi",
            "WSGI module check"
        )

    def test_file_permissions(self):
        """Test file and directory permissions"""
        dirs_to_check = [
            '/var/log/intellivision',
            '/var/log/intellivision/api',
            '/var/log/intellivision/celery',
            '/var/log/intellivision/security',
            '/var/log/intellivision/performance',
            '/var/www/intellivision/media',
            '/var/www/intellivision/media/outputs'
        ]

        all_passed = True
        output = []

        for dir_path in dirs_to_check:
            try:
                # Check if directory exists
                if not os.path.exists(dir_path):
                    output.append(f"✗ {dir_path}: Directory does not exist")
                    all_passed = False
                    continue

                # Check ownership
                stat = os.stat(dir_path)
                if stat.st_uid != os.getuid():
                    output.append(f"✗ {dir_path}: Wrong ownership")
                    all_passed = False
                    continue

                # Check permissions
                if oct(stat.st_mode)[-3:] != '755':
                    output.append(f"✗ {dir_path}: Wrong permissions")
                    all_passed = False
                    continue

                output.append(f"✓ {dir_path}: All checks passed")
            except Exception as e:
                all_passed = False
                output.append(f"✗ {dir_path}: Error during check - {str(e)}")

        return all_passed, "\n".join(output)

    def check_celery_worker(self):
        """Check if Celery worker is running"""
        return self.run_command(
            "systemctl status celery",
            "Celery worker check"
        )

    def check_redis_connection(self):
        """Verify Redis connection"""
        return self.run_command(
            "redis-cli ping",
            "Redis connection check"
        )

    def test_static_files(self):
        """Test if static files are being served by Apache"""
        return self.run_command(
            "curl -I http://localhost/static/admin/css/base.css",
            "Static files check"
        )

    def check_ssl_config(self):
        """Check SSL configuration if enabled"""
        return self.run_command(
            "apache2ctl -M | grep ssl",
            "SSL module check"
        )

    def generate_report(self):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        duration = self.end_time - self.start_time

        with open('test_reports/vm_test_summary.txt', 'w') as f:
            f.write(f"VM Environment Test Summary\n")
            f.write(f"=========================\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Duration: {duration:.2f} seconds\n\n")

            for test_type, result in self.test_results.items():
                f.write(f"{test_type.upper()}:\n")
                f.write(f"Status: {'PASSED' if result['success'] else 'FAILED'}\n")
                if not result['success']:
                    f.write(f"Error Output:\n{result['output']}\n")
                f.write("-" * 50 + "\n\n")

    def run_all_tests(self):
        self.start_time = time.time()

        try:
            # Run VM environment tests
            test_types = [
                ('apache_status', self.check_apache_status),
                ('apache_logs', self.check_apache_logs),
                ('wsgi_config', self.check_wsgi_config),
                ('file_permissions', self.test_file_permissions),
                ('celery_worker', self.check_celery_worker),
                ('redis_connection', self.check_redis_connection),
                ('static_files', self.test_static_files),
                ('ssl_config', self.check_ssl_config)
            ]

            for test_type, test_func in test_types:
                success, output = test_func()
                self.test_results[test_type] = {
                    'success': success,
                    'output': output
                }

        finally:
            self.end_time = time.time()
            self.generate_report()

def main():
    # Check if running as root (required for some system checks)
    if os.geteuid() != 0:
        logger.error("This script must be run as root to perform system checks")
        print("\nUsage: sudo python3 test_in_vm.py")
        sys.exit(1)

    runner = VMTestRunner()
    try:
        runner.run_all_tests()
        logger.info("All tests completed. Check test_reports directory for detailed results.")

        # Print summary
        print("\nTest Summary:")
        print("============")
        passed = sum(1 for result in runner.test_results.values() if result['success'])
        total = len(runner.test_results)
        print(f"Passed: {passed}/{total} tests")

        if passed < total:
            print("\nFailed Tests:")
            for test_type, result in runner.test_results.items():
                if not result['success']:
                    print(f"- {test_type}")

    except Exception as e:
        logger.error(f"Error during test execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
