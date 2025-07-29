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

class TestRunner:
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None

        # Create reports directory if it doesn't exist
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

    def install_dependencies(self):
        logger.info("Installing test dependencies...")
        success, output = self.run_command(
            "pip install -r test-requirements.txt",
            "dependency installation"
        )
        if not success:
            logger.error("Failed to install dependencies. Aborting.")
            sys.exit(1)

    def run_unit_tests(self):
        return self.run_command(
            "pytest -v -m 'unit' --junitxml=test_reports/unit.xml",
            "unit tests"
        )

    def run_integration_tests(self):
        return self.run_command(
            "pytest -v -m 'integration' --junitxml=test_reports/integration.xml",
            "integration tests"
        )

    def run_api_tests(self):
        return self.run_command(
            "pytest -v -m 'api' --junitxml=test_reports/api.xml",
            "API tests"
        )

    def run_concurrent_tests(self):
        return self.run_command(
            "pytest -v -m 'concurrent' --junitxml=test_reports/concurrent.xml",
            "concurrency tests"
        )

    def run_e2e_tests(self):
        return self.run_command(
            "pytest -v -m 'e2e' --junitxml=test_reports/e2e.xml",
            "end-to-end tests"
        )

    def run_load_tests(self):
        return self.run_command(
            "locust --headless -f load_tests/locustfile.py --users 50 --spawn-rate 5 -t 5m --html test_reports/load_test.html",
            "load tests"
        )

    def generate_coverage_report(self):
        return self.run_command(
            "coverage combine && coverage html -d test_reports/coverage",
            "coverage report generation"
        )

    def run_all_tests(self):
        self.start_time = time.time()

        # Install dependencies first
        self.install_dependencies()

        # Run different types of tests
        test_types = [
            ('unit', self.run_unit_tests),
            ('integration', self.run_integration_tests),
            ('api', self.run_api_tests),
            ('concurrent', self.run_concurrent_tests),
            ('e2e', self.run_e2e_tests),
            ('load', self.run_load_tests)
        ]

        for test_type, test_func in test_types:
            success, output = test_func()
            self.test_results[test_type] = {
                'success': success,
                'output': output
            }

        # Generate coverage report
        self.generate_coverage_report()

        self.end_time = time.time()
        self.generate_summary_report()

    def generate_summary_report(self):
        duration = self.end_time - self.start_time
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open('test_reports/summary.txt', 'w') as f:
            f.write(f"Test Summary Report\n")
            f.write(f"==================\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Duration: {duration:.2f} seconds\n\n")

            for test_type, result in self.test_results.items():
                f.write(f"{test_type.upper()} Tests:\n")
                f.write(f"Status: {'PASSED' if result['success'] else 'FAILED'}\n")
                f.write("-" * 50 + "\n\n")

            f.write("\nDetailed test reports and coverage information can be found in the test_reports directory.")

def main():
    runner = TestRunner()
    try:
        runner.run_all_tests()
        logger.info("All tests completed. Check test_reports directory for detailed results.")
    except Exception as e:
        logger.error(f"Error during test execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
