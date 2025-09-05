#!/usr/bin/env python3
# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Test runner for CCCL examples.

This module automatically discovers and runs all example scripts from both
cooperative and parallel directories to ensure they execute without errors.
"""

import importlib
import inspect
import sys
import traceback
from pathlib import Path


def discover_examples():
    """Automatically discover all example files and their functions."""
    tests_dir = Path(__file__).parent
    examples = []

    # Look for examples in both cooperative and parallel directories
    example_directories = ["cooperative/examples", "parallel/examples"]

    for example_dir in example_directories:
        example_path = tests_dir / example_dir
        if not example_path.exists():
            continue

        # Find all Python files in subdirectories
        for python_file in example_path.rglob("*.py"):
            if (
                python_file.name == "__init__.py"
                or python_file.name == "test_examples.py"
            ):
                continue

            # Calculate the relative path from the tests directory
            rel_path = python_file.relative_to(tests_dir)

            # Convert path to module name (e.g., "cooperative/examples/block/reduce.py" -> "cooperative.examples.block.reduce")
            module_name = str(rel_path.with_suffix("")).replace("/", ".")

            # Extract category info for display
            parts = rel_path.parts
            if len(parts) >= 3:
                # e.g., cooperative/examples/block/reduce.py
                framework = parts[0].title()  # Cooperative or Parallel
                category = parts[2].title()  # Block, Warp, Reduction, etc.
                filename = parts[3].replace(".py", "").replace("_", " ").title()
                display_name = f"{framework} - {category} - {filename}"
            elif len(parts) >= 2:
                # e.g., cooperative/examples/reduce.py
                framework = parts[0].title()
                filename = parts[-1].replace(".py", "").replace("_", " ").title()
                display_name = f"{framework} - {filename}"
            else:
                display_name = rel_path.stem.replace("_", " ").title()

            examples.append((display_name, module_name))

    return sorted(examples)


def run_example_module(module_name, display_name):
    """Run all example functions from a module."""
    try:
        print(f"Testing {display_name}...")

        # Import the module
        module = importlib.import_module(module_name)

        # Check if module has a main function - if so, run it
        if hasattr(module, "__main__") or hasattr(module, "main"):
            # Call main if it exists
            if hasattr(module, "main"):
                module.main()
            else:
                # Try to run the module as if it were called directly
                exec(f"import {module_name}; {module_name}.__main__()")
        else:
            # Find and run all example functions (those ending with _example)
            example_functions = []
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isfunction(obj)
                    and name.endswith("_example")
                    and not name.startswith("_")
                ):
                    example_functions.append((name, obj))

            if example_functions:
                for func_name, func in sorted(example_functions):
                    print(f"  Running {func_name}...")
                    func()
            else:
                # If no example functions found, try to run the module directly
                # by checking if it has a __name__ == "__main__" block
                print(f"  Running {module_name} as script...")
                import os
                import subprocess

                module_file = module.__file__
                if module_file:
                    # Run the module as a script
                    result = subprocess.run(
                        [sys.executable, module_file],
                        capture_output=True,
                        text=True,
                        cwd=os.path.dirname(module_file),
                    )
                    if result.returncode != 0:
                        raise Exception(f"Module execution failed: {result.stderr}")
                    print(f"  Output: {result.stdout.strip()}")

        print(f"✓ {display_name} examples passed")
        return True

    except Exception as e:
        print(f"✗ {display_name} examples failed: {e}")
        traceback.print_exc()
        return False


# Create pytest-compatible test functions dynamically
def create_test_functions():
    """Create pytest-compatible test functions for each discovered example."""
    examples = discover_examples()

    for display_name, module_name in examples:
        # Create a test function name from the module name
        test_name = f"test_{module_name.replace('.', '_')}"

        # Create the test function
        def make_test_func(mod_name, disp_name):
            def test_func():
                assert run_example_module(mod_name, disp_name)

            return test_func

        # Add the test function to the global namespace
        globals()[test_name] = make_test_func(module_name, display_name)
        globals()[test_name].__name__ = test_name
        globals()[test_name].__doc__ = f"Test {display_name} examples"


# Create test functions for pytest
create_test_functions()
