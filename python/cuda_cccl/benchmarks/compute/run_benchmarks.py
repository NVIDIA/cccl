#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Run Python cuda.compute and C++ CUB benchmarks.

Prerequisites: C++ benchmarks must be built first via:
  cd /path/to/cccl && ./ci/build_cub.sh -arch 89

Usage:
  python run_benchmarks.py [options]

Options:
  -d, --device ID       GPU device ID [default: 0]
  -b, --benchmark NAME  Run specific benchmark only [default: all]
  --py                  Only run Python benchmarks
  --cpp                 Only run C++ benchmarks
  --quick, -q           Run with reduced parameter set for fast testing
  -h, --help            Show this help message

Benchmark names follow CUB structure:
  e.g. transform/fill, transform/babelstream

Examples:
  python run_benchmarks.py                              # Run all benchmarks
  python run_benchmarks.py -b transform/fill            # Only fill benchmark
  python run_benchmarks.py -b reduce/sum -d 0           # Reduce sum on device 0
  python run_benchmarks.py -b scan/exclusive/sum --py   # Only Python
  python run_benchmarks.py --quick                      # Quick mode (reduced params)
  python run_benchmarks.py -b merge_sort/keys -q        # Quick mode, single benchmark
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
CCCL_ROOT = SCRIPT_DIR.parents[3]
RESULTS_DIR = SCRIPT_DIR / "results"
CUB_BENCH_DIR = CCCL_ROOT / "build" / "cub" / "bin"
QUICK_CONFIG_FILE = SCRIPT_DIR / "quick_configs.yaml"

# Supported benchmarks (Python implementations available)
SUPPORTED_BENCHMARKS = [
    "transform/fill",
    "transform/babelstream",
    "transform/heavy",
    "transform/fib",
    "transform/grayscale",
    "transform/complex_cmp",
    "transform_reduce/sum",
    "reduce/sum",
    "reduce/min",
    "reduce/custom",
    "reduce/deterministic",
    "reduce/nondeterministic",
    "scan/exclusive/sum",
    "scan/exclusive/custom",
    "histogram/even",
    "select/if",
    "select/flagged",
    "select/unique_by_key",
    "radix_sort/keys",
    "radix_sort/pairs",
    "merge_sort/keys",
    "merge_sort/pairs",
    "segmented_sort/keys",
    "segmented_reduce/sum",
    "segmented_reduce/custom",
    "partition/three_way",
]

# Axes that use power-of-two values (need [pow2] suffix for nvbench)
# These are the base names (without {ct}/{io} suffixes)
POW2_AXES_CPP = {"Elements", "MaxSegSize", "MaxSegmentSize", "SegmentSize", "Segments"}
POW2_AXES_PY = {"Elements", "MaxSegSize", "MaxSegmentSize", "Segments"}

# Axis name mappings from C++ to Python
# C++ uses suffixes like {ct} (compile-time) and {io} (input/output)
# Python uses simple names
# Some benchmarks also have different axis names entirely
CPP_TO_PY_AXIS_MAP = {
    # Common mappings
    "T{ct}": "T",
    "Elements{io}": "Elements",
    "KeyT{ct}": "KeyT",
    "ValueT{ct}": "ValueT",
    # Histogram uses different name
    "SampleT{ct}": "SampleT",
}


def strip_axis_suffix(axis_name: str) -> str:
    """Strip {ct} or {io} suffix from axis name for Python benchmarks.

    e.g., "T{ct}" -> "T", "Elements{io}" -> "Elements"
    Also handles special mappings like "SampleT{ct}" -> "T"
    """
    if axis_name in CPP_TO_PY_AXIS_MAP:
        return CPP_TO_PY_AXIS_MAP[axis_name]
    # Generic suffix stripping for any other axes
    if axis_name.endswith("{ct}") or axis_name.endswith("{io}"):
        return axis_name.rsplit("{", 1)[0]
    return axis_name


def get_base_axis_name(axis_name: str) -> str:
    """Get base axis name (without suffix) for POW2 check.

    e.g., "Elements{io}" -> "Elements"
    """
    if axis_name.endswith("{ct}") or axis_name.endswith("{io}"):
        return axis_name.rsplit("{", 1)[0]
    return axis_name


# ============================================================================
# Helper Functions
# ============================================================================


def print_banner(msg: str) -> None:
    """Print a banner message."""
    print()
    print("=" * 72)
    print(msg)
    print("=" * 72)
    print()


def print_section(msg: str) -> None:
    """Print a section header."""
    print()
    print("-" * 72)
    print(msg)
    print("-" * 72)


def load_quick_configs() -> dict:
    """Load quick mode configurations from YAML file."""
    with open(QUICK_CONFIG_FILE) as f:
        return yaml.safe_load(f)


def build_axis_args_from_config(axis_config: dict, for_python: bool) -> list:
    """Build --axis arguments for nvbench CLI from an axis config dict.

    Args:
        axis_config: Dict of axis_name -> value
        for_python: If True, strip C++ suffixes for Python benchmarks
    """
    args = []
    for axis_name, value in axis_config.items():
        # For Python, strip C++ suffixes from axis names
        if for_python:
            axis_name = strip_axis_suffix(axis_name)

        # Check if this is a power-of-two axis (using base name)
        base_name = get_base_axis_name(axis_name)
        if for_python and base_name == "SegmentSize":
            actual_value = 2 ** int(value)
            args.extend(["--axis", f"{axis_name}={actual_value}"])
            continue

        pow2_axes = POW2_AXES_PY if for_python else POW2_AXES_CPP
        if base_name in pow2_axes:
            args.extend(["--axis", f"{axis_name}[pow2]={value}"])
        else:
            args.extend(["--axis", f"{axis_name}={value}"])
    return args


def get_quick_config_entry(benchmark: str, quick_configs: dict) -> dict:
    """Get quick config entry for a benchmark, raising if missing."""
    if benchmark not in quick_configs:
        raise ValueError(
            f"Benchmark '{benchmark}' not found in quick_configs.yaml.\n"
            f"Cannot run in --quick mode. Add configuration for this benchmark."
        )
    return quick_configs[benchmark]


def get_cpp_binary(benchmark: str) -> str:
    """Get C++ binary name from benchmark path.

    e.g., "transform/fill" -> "cub.bench.transform.fill.base"
    """
    return f"cub.bench.{benchmark.replace('/', '.')}.base"


def get_py_script(benchmark: str) -> Path:
    """Get Python script path from benchmark path.

    e.g., "transform/fill" -> "transform/fill.py"
    """
    return SCRIPT_DIR / f"{benchmark}.py"


def get_result_path(benchmark: str, suffix: str) -> Path:
    """Get results file path.

    e.g., "transform/fill", "cpp" -> "results/transform/fill_cpp.json"
    """
    bench_path = Path(benchmark)
    return RESULTS_DIR / bench_path.parent / f"{bench_path.name}_{suffix}.json"


def get_log_path(benchmark: str, suffix: str) -> Path:
    """Get log file path under results/logs.

    e.g., "transform/fill", "cpp" -> "results/logs/transform/fill_cpp.log"
    """
    bench_path = Path(benchmark)
    return RESULTS_DIR / "logs" / bench_path.parent / f"{bench_path.name}_{suffix}.log"


def run_and_log(cmd: list, log_path: Path) -> None:
    """Run command and write stdout/stderr to log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Command: {shlex.join(cmd)}\n\n")
        log_file.flush()
        subprocess.run(cmd, check=True, stdout=log_file, stderr=log_file)


# ============================================================================
# Benchmark Runner
# ============================================================================


def run_benchmark(
    benchmark: str,
    device: str,
    run_py: bool,
    run_cpp: bool,
    quick_mode: bool,
    quick_configs: dict,
) -> dict:
    """Run a single benchmark.

    Returns dict with paths to generated result files.
    """
    cpp_binary = get_cpp_binary(benchmark)
    py_script = get_py_script(benchmark)
    cpp_result = get_result_path(benchmark, "cpp")
    py_result = get_result_path(benchmark, "py")
    cpp_log = get_log_path(benchmark, "cpp")
    py_log = get_log_path(benchmark, "py")

    # Ensure results subdirectory exists
    cpp_result.parent.mkdir(parents=True, exist_ok=True)

    # Build axis arguments for quick mode
    cpp_axis_args = []
    py_axis_args = []
    if quick_mode:
        config_entry = get_quick_config_entry(benchmark, quick_configs)
        if "benchmarks" in config_entry:
            for bench_name, axis_config in config_entry["benchmarks"].items():
                cpp_axis_args.extend(["--benchmark", bench_name])
                cpp_axis_args.extend(
                    build_axis_args_from_config(axis_config, for_python=False)
                )
                py_axis_args.extend(["--benchmark", bench_name])
                py_axis_args.extend(
                    build_axis_args_from_config(axis_config, for_python=True)
                )
        else:
            cpp_axis_args = build_axis_args_from_config(config_entry, for_python=False)
            py_axis_args = build_axis_args_from_config(config_entry, for_python=True)

    results = {}

    # Run C++ benchmark
    if run_cpp:
        print(f"Running C++ benchmark: {cpp_binary}")

        cpp_bin = CUB_BENCH_DIR / cpp_binary
        if not cpp_bin.exists():
            print(f"ERROR: C++ binary not found: {cpp_bin}")
            print()
            print("Please build C++ benchmarks first:")
            print(f"  cd {CCCL_ROOT}")
            print("  ./ci/build_cub.sh -arch <your_gpu_arch>  # e.g., 89 for RTX 4090")
            print()
            print("Available benchmarks in build directory:")
            if CUB_BENCH_DIR.exists():
                binaries = list(CUB_BENCH_DIR.glob("*.base"))[:10]
                for b in binaries:
                    print(f"  {b.name}")
                if len(list(CUB_BENCH_DIR.glob("*.base"))) > 10:
                    print("  ...")
            else:
                print(f"  (directory not found: {CUB_BENCH_DIR})")
            sys.exit(1)

        cmd = [str(cpp_bin), "--json", str(cpp_result), "--devices", device]
        cmd.extend(cpp_axis_args)
        run_and_log(cmd, cpp_log)
        print(f"  Results: {cpp_result}")
        print(f"  Log: {cpp_log}")
        results["cpp"] = cpp_result

    # Run Python benchmark
    if run_py:
        print(f"Running Python benchmark: {py_script.relative_to(SCRIPT_DIR)}")

        if not py_script.exists():
            print(f"ERROR: Python script not found: {py_script}")
            sys.exit(1)

        cmd = [
            sys.executable,
            str(py_script),
            "--json",
            str(py_result),
            "--devices",
            device,
        ]
        cmd.extend(py_axis_args)
        run_and_log(cmd, py_log)
        print(f"  Results: {py_result}")
        print(f"  Log: {py_log}")
        results["py"] = py_result

    return results


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run Python cuda.compute and C++ CUB benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported benchmarks:
{chr(10).join(f"  {b}" for b in SUPPORTED_BENCHMARKS)}
""",
    )
    parser.add_argument(
        "-d", "--device", default="0", help="GPU device ID [default: 0]"
    )
    parser.add_argument(
        "-b", "--benchmark", help="Run specific benchmark only [default: all]"
    )
    parser.add_argument("--py", action="store_true", help="Only run Python benchmarks")
    parser.add_argument("--cpp", action="store_true", help="Only run C++ benchmarks")
    parser.add_argument(
        "-q",
        "--quick",
        action="store_true",
        help="Run with reduced parameter set for fast testing",
    )
    args = parser.parse_args()

    # Determine what to run
    run_py = True
    run_cpp = True
    if args.py and not args.cpp:
        run_cpp = False
    elif args.cpp and not args.py:
        run_py = False

    # Load quick configs if needed
    quick_configs = {}
    if args.quick:
        quick_configs = load_quick_configs()

    # Validate and determine benchmarks to run
    if args.benchmark:
        if args.benchmark not in SUPPORTED_BENCHMARKS:
            print(f"ERROR: Benchmark '{args.benchmark}' not supported.")
            print()
            print("Available benchmarks:")
            for b in SUPPORTED_BENCHMARKS:
                print(f"  {b}")
            sys.exit(1)
        benchmarks_to_run = [args.benchmark]
    else:
        benchmarks_to_run = SUPPORTED_BENCHMARKS

    # Print configuration
    print_banner("CCCL Benchmark Runner")

    print("Configuration:")
    print(f"  CCCL Root:      {CCCL_ROOT}")
    print(f"  C++ Binaries:   {CUB_BENCH_DIR}")
    print(f"  Results Dir:    {RESULTS_DIR}")
    print(f"  Device:         {args.device}")
    print(f"  Benchmarks:     {' '.join(benchmarks_to_run)}")
    print(f"  Run C++:        {run_cpp}")
    print(f"  Run Python:     {run_py}")
    print(f"  Quick Mode:     {args.quick}")
    print()

    # Check C++ binaries directory exists (if running C++)
    if run_cpp and not CUB_BENCH_DIR.exists():
        print(f"ERROR: C++ benchmark directory not found: {CUB_BENCH_DIR}")
        print()
        print("Please build C++ benchmarks first:")
        print(f"  cd {CCCL_ROOT}")
        print("  ./ci/build_cub.sh -arch <your_gpu_arch>  # e.g., 89 for RTX 4090")
        sys.exit(1)

    # Run benchmarks
    all_results = {}
    for bench in benchmarks_to_run:
        print_section(f"Benchmark: {bench}")
        results = run_benchmark(
            bench, args.device, run_py, run_cpp, args.quick, quick_configs
        )
        all_results[bench] = results
        print()

    # Print summary
    print_banner("Summary")

    print(f"Results directory: {RESULTS_DIR}")
    print()
    print("Generated files:")
    for bench in benchmarks_to_run:
        cpp_result = get_result_path(bench, "cpp")
        py_result = get_result_path(bench, "py")
        cpp_log = get_log_path(bench, "cpp")
        py_log = get_log_path(bench, "py")
        print(f"  {bench}:")
        if cpp_result.exists():
            print(f"    C++ results:    {cpp_result}")
        if cpp_log.exists():
            print(f"    C++ log:        {cpp_log}")
        if py_result.exists():
            print(f"    Python results: {py_result}")
        if py_log.exists():
            print(f"    Python log:     {py_log}")

    print()
    print("To compare results, run:")
    for bench in benchmarks_to_run:
        print(f"  python analysis/python_vs_cpp_summary.py -b {bench} -d {args.device}")
    print()


if __name__ == "__main__":
    main()
