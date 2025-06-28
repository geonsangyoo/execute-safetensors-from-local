#!/usr/bin/env python3
"""
Comprehensive linting script for the project.
Runs isort, black, flake8, and mypy to ensure code quality.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"\n{'='*50}")
    print(f"Running {description}...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        success = result.returncode == 0
        if success:
            print(f"✅ {description} completed successfully!")
        else:
            print(f"❌ {description} failed with return code {result.returncode}")

        return success, result.stdout + result.stderr

    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False, str(e)


def main() -> None:
    """Main linting function."""
    print("🚀 Starting comprehensive code linting...")

    # Get Python files to lint
    python_files = list(Path(".").glob("*.py"))
    python_files.extend(list(Path("src").glob("**/*.py")))
    python_files.extend(list(Path("tools").glob("**/*.py")))

    if not python_files:
        print("❌ No Python files found in current directory")
        sys.exit(1)

    print(f"📁 Found {len(python_files)} Python files to lint")

    all_success = True
    results = {}

    # 1. Run isort to sort imports (uses .isort.cfg)
    isort_cmd = (
        ["isort", "--check-only", "--diff", "."]
        if "--check-only" in sys.argv
        else ["isort", "."]
    )
    success, output = run_command(isort_cmd, "isort (import sorting)")
    results["isort"] = (success, output)
    all_success = all_success and success

    # 2. Run black to format code (uses .black)
    black_cmd = (
        ["black", "--check", "--diff", "."]
        if "--check-only" in sys.argv
        else ["black", "."]
    )
    success, output = run_command(black_cmd, "black (code formatting)")
    results["black"] = (success, output)
    all_success = all_success and success

    # 3. Run flake8 for style checking (uses .flake8)
    success, output = run_command(
        ["flake8", ".", "--config=.flake8"], "flake8 (style checking)"
    )
    results["flake8"] = (success, output)
    all_success = all_success and success

    # 4. Run mypy for type checking (uses mypy.ini)
    success, output = run_command(
        ["mypy", ".", "--config-file=mypy.ini"], "mypy (type checking)"
    )
    results["mypy"] = (success, output)
    all_success = all_success and success

    # Summary
    print(f"\n{'='*60}")
    print("📊 LINTING SUMMARY")
    print("=" * 60)

    for tool, (success, output) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{tool:10} : {status}")

    if all_success:
        print("\n🎉 All linting checks passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some linting checks failed. Please fix the issues above.")
        if "--check-only" not in sys.argv:
            print("💡 Run without --check-only to automatically fix formatting issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
