#!/usr/bin/env python3
"""
Regression tests for orchestrator improvements.
Tests multi-block parsing, natural language aliases, default handling, and error cases.
"""
import subprocess
import json
import tempfile
import os
import sys
from pathlib import Path

def test_orchestrator_with_goal(goal, description):
    """Test the orchestrator with a specific goal and return results"""
    print(f"\n🧪 Testing: {description}")
    print(f"   Goal: {goal[:80]}{'...' if len(goal) > 80 else ''}")

    # Create a temporary script that uses --goal
    test_script = f'''
import sys
sys.path.append('.')
from orch.orchestrator import main
import json
result = main({goal!r})
print(json.dumps(result, ensure_ascii=False))
'''

    tmp_path = Path(tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False).name)
    tmp_path.write_text(test_script)
    tmp_path.chmod(0o755)

    try:
        result = subprocess.run([sys.executable, str(tmp_path)],
                             capture_output=True, text=True, timeout=30, cwd='.')

        output = result.stdout.strip()
        error = result.stderr.strip()

        if output:
            try:
                parsed_result = json.loads(output)
                success = parsed_result.get("done", False)
                log = parsed_result.get("log", [])

                print(f"   {'✅ PASS' if success else '❌ FAIL'}: {len(log)} steps executed")

                if error:
                    print(f"   stderr: {error}")

                return success, len(log), error

            except json.JSONDecodeError as e:
                print(f"   ❌ FAIL: Invalid JSON output - {e}")
                print(f"   stdout: {output}")
                return False, 0, error
        else:
            print("   ❌ FAIL: No output received")
            if error:
                print(f"   stderr: {error}")
            return False, 0, error

    except subprocess.TimeoutExpired:
        print("   ❌ FAIL: Test timed out")
        return False, 0, "timeout"
    except Exception as e:
        print(f"   ❌ FAIL: Exception - {e}")
        return False, 0, str(e)
    finally:
        tmp_path.unlink(missing_ok=True)

def run_regression_tests():
    """Run all regression tests"""
    print("🚀 Running Orchestrator Regression Tests")
    print("=" * 50)

    test_cases = [
        # Basic functionality tests
        ("Create hello.txt and list current directory",
         "Basic file creation and directory listing"),

        ("touch test.txt", "Natural language touch command"),

        ("ls", "Natural language ls command"),

        ("show contents of .", "Natural language directory contents"),

        ("cat hello.txt", "Natural language cat command"),

        # Multi-block scenarios (should work with improved parsing)
        ("Create test1.txt and test2.txt then list directory",
         "Multi-step operation"),

        ("touch file1.txt && touch file2.txt && ls",
         "Shell-like command chaining"),

        # Error handling tests
        ("use unknown_tool", "Invalid tool should be handled gracefully"),

        ("read_file {\"invalid\": \"json\"}", "Invalid JSON should be handled"),

        ("list_dir {\"path\": \"/nonexistent/path/that/should/fail\"}",
         "Path error should be handled gracefully"),

        # Default argument tests
        ("list_dir", "Should default to current directory"),

        ("list_dir {}", "Should default to current directory with empty dict"),

        # Natural language variations
        ("please create a file called natural.txt", "Natural language file creation"),

        ("show me what's in this folder", "Natural language directory browsing"),

        ("open the README.md file", "Natural language file reading"),

        # Edge cases
        ("<tool>list_dir</tool><args>{}</args>", "Direct tool format with empty args"),

        ("ls .", "List current directory explicitly"),

        ("touch /tmp/test_nonexistent.txt", "File creation with absolute path"),
    ]

    results = []
    total_tests = len(test_cases)
    passed = 0

    for i, (goal, description) in enumerate(test_cases, 1):
        print(f"\n[{i}/{total_tests}] ", end='')
        success, steps, error = test_orchestrator_with_goal(goal, description)
        results.append((success, steps, error))

        if success:
            passed += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total_tests} passed ({passed/total_tests*100:.1f}%)")

    if passed == total_tests:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️ Some tests failed. Check output above for details.")
        # Show failed tests
        print("\n❌ Failed tests:")
        for i, (success, steps, error) in enumerate(results):
            if not success:
                goal, desc = test_cases[i]
                print(f"  - {desc}: {goal}")
                if error:
                    print(f"    Error: {error}")
        return False

if __name__ == "__main__":
    success = run_regression_tests()
    sys.exit(0 if success else 1)
