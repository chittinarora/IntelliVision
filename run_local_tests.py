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

class LocalTestRunner:
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.manage_py = os.path.join('intellivision', 'manage.py')
        os.makedirs('test_reports', exist_ok=True)

        # Set up environment variables
        os.environ['PYTHONPATH'] = 'intellivision'
        os.environ['DJANGO_SETTINGS_MODULE'] = 'intellivision.settings'

    def run_command(self, command, description):
        logger.info(f"Running {description}...")
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = 'intellivision'
            env['DJANGO_SETTINGS_MODULE'] = 'intellivision.settings'

            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Error in {description}: {e.stderr}")
            return False, e.stderr

    def run_unit_tests(self):
        """Run Django unit tests"""
        return self.run_command(
            f"cd intellivision && python3 manage.py test intellivision.apps.video_analytics.tests --verbosity=2",
            "video analytics unit tests"
        )

    def run_face_auth_tests(self):
        """Run face authentication tests"""
        return self.run_command(
            f"cd intellivision && python3 manage.py test intellivision.apps.face_auth.tests --verbosity=2",
            "face auth tests"
        )

    def test_file_permissions(self):
        """Test file and directory permissions"""
        dirs_to_check = [
            'logs',
            'logs/api',
            'logs/celery',
            'logs/security',
            'logs/performance',
            'media',
            'media/outputs'
        ]

        all_passed = True
        output = []

        for dir_path in dirs_to_check:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                output.append(f"Created missing directory: {dir_path}")

            try:
                # Create a test file
                test_file = os.path.join(dir_path, 'test_write.tmp')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                output.append(f"✓ {dir_path}: Write test passed")
            except Exception as e:
                all_passed = False
                output.append(f"✗ {dir_path}: Write test failed - {str(e)}")

        return all_passed, "\n".join(output)

    def check_celery_config(self):
        """Verify Celery configuration"""
        try:
            # Add intellivision directory to Python path
            sys.path.insert(0, 'intellivision')
            from celery_app import app
            i = app.control.inspect()
            return True, "Celery configuration valid"
        except Exception as e:
            return False, f"Celery configuration error: {str(e)}"
        finally:
            # Remove from path
            if 'intellivision' in sys.path:
                sys.path.remove('intellivision')

    def verify_dependencies(self):
        """Check if all required dependencies are installed"""
        return self.run_command(
            "pip check",
            "dependency verification"
        )

    def generate_report(self):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        duration = self.end_time - self.start_time

        with open('test_reports/local_test_summary.txt', 'w') as f:
            f.write(f"Local Test Summary\n")
            f.write(f"=================\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Duration: {duration:.2f} seconds\n\n")

            for test_type, result in self.test_results.items():
                f.write(f"{test_type.upper()}:\n")
                f.write(f"Status: {'PASSED' if result['success'] else 'FAILED'}\n")
                if not result['success']:
                    f.write(f"Error Output:\n{result['output']}\n")
                f.write("-" * 50 + "\n\n")

            # Add deployment checklist
            f.write("\nVM/Apache Deployment Checklist\n")
            f.write("==========================\n")
            f.write("□ Verify Apache configuration in VM\n")
            f.write("□ Check Apache log paths match VM setup\n")
            f.write("□ Verify file permissions in VM environment\n")
            f.write("□ Test media file uploads in VM\n")
            f.write("□ Verify Celery worker starts correctly in VM\n")
            f.write("□ Check Redis connection in VM environment\n")
            f.write("□ Verify static files are served by Apache\n")
            f.write("□ Test WSGI configuration\n")
            f.write("□ Verify SSL/TLS configuration if applicable\n")
            f.write("□ Check Apache error logs after deployment\n")

    def run_all_tests(self):
        self.start_time = time.time()

        try:
            # Run core functionality tests
            test_types = [
                ('unit', self.run_unit_tests),
                ('face_auth', self.run_face_auth_tests),
                ('permissions', self.test_file_permissions),
                ('celery', self.check_celery_config),
                ('dependencies', self.verify_dependencies)
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
    runner = LocalTestRunner()
    try:
        runner.run_all_tests()
        logger.info("All tests completed. Check test_reports directory for detailed results.")

        # Print deployment reminder
        print("\nIMPORTANT DEPLOYMENT REMINDER:")
        print("==============================")
        print("1. These tests check core functionality only")
        print("2. Review test_reports/local_test_summary.txt for VM/Apache deployment checklist")
        print("3. Additional testing required in VM environment")
        print("4. Monitor Apache logs after deployment")

    except Exception as e:
        logger.error(f"Error during test execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
