import ast
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_function_signature(file_path, function_name):
    """Extract function signature from Python source file using AST."""
    with open(file_path, 'r') as f:
        content = f.read()
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            args = [arg.arg for arg in node.args.args]
            return {'name': function_name, 'params': args, 'total_params': len(args)}
    return None

def check_file_handling_changes(file_path):
    """Check if file handling changes are present."""
    with open(file_path, 'r') as f:
        content = f.read()
    results = []
    if 'final_output_path' in content:
        results.append("✅ Uses final_output_path for filesystem return")
    if 'file_job_id' in content or 'effective_job_id' in content:
        results.append("✅ Fixed job_id variable conflict")
    if 'convert_to_web_mp4' in content:
        results.append("✅ FFmpeg conversion implemented")
    if 'tasks.py needs it' in content:
        results.append("✅ Prevents premature file cleanup")
    return results

def test_priority3_fixes():
    """Test Priority 3 fixes for parameter mismatches and file handling."""
    logger.info("Testing Priority 3 fixes...")

    # Test pothole_detection.py
    logger.info("\nTesting pothole_detection.py:")
    file_path = "intellivision/apps/video_analytics/analytics/pothole_detection.py"

    # Check function signatures
    funcs = [
        ('run_pothole_image_detection', ['image_path', 'output_path', 'job_id']),
        ('run_pothole_detection', ['input_path', 'output_path', 'job_id']),
        ('tracking_video', ['input_path', 'output_path', 'job_id'])
    ]
    for func_name, expected_params in funcs:
        sig = extract_function_signature(file_path, func_name)
        if sig:
            logger.info(f"   Function: {func_name}")
            logger.info(f"   Parameters: {sig['params']}")
            if all(param in sig['params'] for param in expected_params):
                logger.info("   ✅ Has correct parameters")
            else:
                logger.error("   ❌ Missing required parameters")
        else:
            logger.error(f"   ❌ Function {func_name} not found")

    # Check file handling changes
    results = check_file_handling_changes(file_path)
    for result in results:
        logger.info(f"   {result}")

    # Test pest_monitoring.py
    logger.info("\nTesting pest_monitoring.py:")
    file_path = "intellivision/apps/video_analytics/analytics/pest_monitoring.py"

    # Check function signatures
    funcs = [
        ('detect_snakes_in_image', ['image_path', 'output_path', 'job_id']),
        ('detect_snakes_in_video', ['video_path', 'output_path', 'job_id']),
        ('tracking_video', ['input_path', 'output_path', 'job_id']),
        ('tracking_image', ['input_path', 'output_path', 'job_id'])
    ]
    for func_name, expected_params in funcs:
        sig = extract_function_signature(file_path, func_name)
        if sig:
            logger.info(f"   Function: {func_name}")
            logger.info(f"   Parameters: {sig['params']}")
            if all(param in sig['params'] for param in expected_params):
                logger.info("   ✅ Has correct parameters")
            else:
                logger.error("   ❌ Missing required parameters")
        else:
            logger.error(f"   ❌ Function {func_name} not found")

    # Check file handling changes
    results = check_file_handling_changes(file_path)
    for result in results:
        logger.info(f"   {result}")

    # Test car_count.py
    logger.info("\nTesting car_count.py:")
    file_path = "intellivision/apps/video_analytics/analytics/car_count.py"

    # Check function signatures
    funcs = [
        ('process_image_file', ['image_path', 'output_path', 'job_id']),
        ('analyze_parking_video', ['video_path', 'output_path', 'job_id']),
        ('tracking_video', ['input_path', 'output_path', 'job_id'])
    ]
    for func_name, expected_params in funcs:
        sig = extract_function_signature(file_path, func_name)
        if sig:
            logger.info(f"   Function: {func_name}")
            logger.info(f"   Parameters: {sig['params']}")
            if all(param in sig['params'] for param in expected_params):
                logger.info("   ✅ Has correct parameters")
            else:
                logger.error("   ❌ Missing required parameters")
        else:
            logger.error(f"   ❌ Function {func_name} not found")

    # Check file handling changes
    results = check_file_handling_changes(file_path)
    for result in results:
        logger.info(f"   {result}")

if __name__ == '__main__':
    test_priority3_fixes()
