#!/usr/bin/env python
"""
Compare Python cuda.compute vs C++ CCCL benchmark performance.

Usage:
    # Compare by benchmark path (looks in results/ directory)
    python python_vs_cpp_summary.py -b transform/fill
    python python_vs_cpp_summary.py -b reduce/sum -d 0
    python python_vs_cpp_summary.py -b scan/exclusive/sum -d 0

    # Compare all supported benchmarks
    python python_vs_cpp_summary.py

    # Legacy: Compare specific files
    python python_vs_cpp_summary.py results/transform/fill_py.json results/transform/fill_cpp.json
"""

import argparse
import math
import sys
from itertools import product
from pathlib import Path

try:
    import tabulate
except ImportError:
    print("Error: tabulate not installed. Run: pip install tabulate")
    sys.exit(1)

import utils

# Supported benchmarks (must match run_benchmarks.sh)
# Format: "category/name" or "category/subcategory/name"
SUPPORTED_BENCHMARKS = [
    "transform/fill",
    "transform/babelstream",
    "transform/heavy",
    "reduce/sum",
    "reduce/min",
    "scan/exclusive/sum",
    "histogram/even",
    "select/if",
    "radix_sort/keys",
    "radix_sort/pairs",
    "segmented_reduce/sum",
    # Add more as implemented:
    # "unique/by_key",
]

# Standard axes that are handled specially (not used for grouping)
# - Elements: always shown as table rows
# - T: grouped by type, shown as section headers
# - OffsetT: filtered to I64 for Python compatibility
STANDARD_AXES = {"Elements", "T", "OffsetT"}

# Axes to filter (Python doesn't have these, prefer specific values for fair comparison)
FILTER_AXES = {
    "OffsetT": "I64",  # Python uses 64-bit offsets
}


def get_result_paths(results_dir: Path, benchmark: str):
    """
    Get paths for Python and C++ result files.

    Args:
        results_dir: Base results directory
        benchmark: Benchmark path like "transform/fill" or "scan/exclusive/sum"

    Returns:
        Tuple of (py_path, cpp_path, comparison_path)
    """
    # Split benchmark path into directory and name
    parts = benchmark.split("/")
    name = parts[-1]
    subdir = "/".join(parts[:-1])

    base_dir = results_dir / subdir
    py_path = base_dir / f"{name}_py.json"
    cpp_path = base_dir / f"{name}_cpp.json"
    comparison_path = base_dir / f"{name}_comparison.txt"

    return py_path, cpp_path, comparison_path


def extract_measurements(results):
    """Extract all state measurements with GPU and CPU time."""
    measurements = []

    for benchmark in results.get("benchmarks", []):
        bench_name = benchmark.get("name", "unknown")
        # Normalize benchmark name: remove "bench_" prefix for compatibility
        if bench_name.startswith("bench_"):
            bench_name = bench_name[6:]

        for state in benchmark.get("states", []):
            if state.get("is_skipped"):
                continue

            # Get axis values
            # Normalize axis names by removing nvbench tags like {io}, {ct}
            axes = {}
            for ax in state.get("axis_values", []):
                name = ax["name"]
                # Remove nvbench tags: Elements{io} → Elements, T{ct} → T
                if "{" in name:
                    name = name.split("{")[0]
                axes[name] = ax["value"]

            # Get GPU and CPU time from summaries
            gpu_time = None
            cpu_time = None
            for summary in state.get("summaries", []):
                summary_name = summary.get("name", "")
                if (
                    "GPU Time" in summary_name
                    and "Min" not in summary_name
                    and "Max" not in summary_name
                ):
                    for data in summary.get("data", []):
                        if data["name"] == "value":
                            gpu_time = float(data["value"])
                            break
                elif summary_name == "CPU Time":  # Mean CPU time
                    for data in summary.get("data", []):
                        if data["name"] == "value":
                            cpu_time = float(data["value"])
                            break

            if gpu_time:
                measurements.append(
                    {
                        "benchmark": bench_name,
                        "device": state["device"],
                        "axes": axes,
                        "gpu_time": gpu_time,
                        "cpu_time": cpu_time,  # May be None
                    }
                )

    return measurements


def summarize_skips(results):
    """Return (total_states, skipped_states, unique_reasons) for an nvbench JSON root."""
    total = 0
    skipped = 0
    reasons = set()
    for benchmark in results.get("benchmarks", []):
        for state in benchmark.get("states", []):
            total += 1
            if state.get("is_skipped"):
                skipped += 1
                reason = state.get("skip_reason")
                if reason:
                    reasons.add(str(reason))
    return total, skipped, reasons


def format_duration(seconds):
    """Format time using nvbench conventions."""
    if seconds >= 1:
        return "%0.3f s" % seconds
    elif seconds >= 1e-3:
        return "%0.3f ms" % (seconds * 1e3)
    elif seconds >= 1e-6:
        return "%0.3f us" % (seconds * 1e6)
    else:
        return "%0.3f us" % (seconds * 1e6)


def format_percentage(value):
    """Format percentage value."""
    return "%0.2f%%" % (value * 100.0)


def filter_measurements(measurements, axis_filters):
    """
    Filter measurements by axis values.

    Args:
        measurements: List of measurement dicts
        axis_filters: Dict of {axis_name: preferred_value}

    Returns:
        Filtered list of measurements
    """
    result = measurements
    for axis_name, preferred_value in axis_filters.items():
        # Check if this axis exists in any measurement
        has_axis = any(axis_name in m["axes"] for m in result)
        if not has_axis:
            continue

        # Filter to preferred value if it exists
        filtered = [m for m in result if m["axes"].get(axis_name) == preferred_value]
        if filtered:
            result = filtered
        # Otherwise keep all (don't filter if preferred value doesn't exist)

    return result


def get_grouping_axes(measurements):
    """
    Identify axes that should be used for hierarchical grouping.

    These are algorithm-specific axes like Entropy, Bins, InPlace that
    affect performance characteristics and should be shown separately.

    Returns:
        List of axis names (sorted for consistent ordering)
    """
    all_axes = set()
    for m in measurements:
        all_axes.update(m["axes"].keys())

    # Grouping axes = all axes except standard ones
    grouping_axes = all_axes - STANDARD_AXES
    return sorted(grouping_axes)


def get_axis_values(measurements, axis_name):
    """Get sorted unique values for an axis."""
    values = set()
    for m in measurements:
        if axis_name in m["axes"]:
            values.add(m["axes"][axis_name])
    return sorted(values)


def filter_by_axes(measurements, axis_values):
    """
    Filter measurements to match specific axis values.

    Args:
        measurements: List of measurement dicts
        axis_values: Dict of {axis_name: value} to match

    Returns:
        Filtered list
    """
    result = []
    for m in measurements:
        match = True
        for axis_name, value in axis_values.items():
            if m["axes"].get(axis_name) != value:
                match = False
                break
        if match:
            result.append(m)
    return result


def print_comparison_table(py_measurements, cpp_measurements, output_fn):
    """Print comparison table for given measurements."""
    # Group by element size
    py_sizes = set(int(m["axes"].get("Elements", 0)) for m in py_measurements)
    cpp_sizes = set(int(m["axes"].get("Elements", 0)) for m in cpp_measurements)
    element_sizes = sorted(py_sizes & cpp_sizes)  # Only sizes in both

    if not element_sizes:
        output_fn("No matching element sizes found.")
        return

    # Build table data with CPU time for interpreter overhead
    table_data = []
    headers = [
        "Elements",
        "C++ GPU",
        "Py GPU",
        "% Slower",
        "C++ CPU",
        "Py CPU",
        "CPU Ovhd",
    ]

    for size in element_sizes:
        py_times = [
            m["gpu_time"]
            for m in py_measurements
            if int(m["axes"].get("Elements", 0)) == size
        ]
        cpp_times = [
            m["gpu_time"]
            for m in cpp_measurements
            if int(m["axes"].get("Elements", 0)) == size
        ]

        py_cpu_times = [
            m["cpu_time"]
            for m in py_measurements
            if int(m["axes"].get("Elements", 0)) == size
            and m.get("cpu_time") is not None
        ]
        cpp_cpu_times = [
            m["cpu_time"]
            for m in cpp_measurements
            if int(m["axes"].get("Elements", 0)) == size
            and m.get("cpu_time") is not None
        ]

        if not py_times or not cpp_times:
            continue

        # Average if multiple measurements (shouldn't happen with proper grouping)
        py_avg = sum(py_times) / len(py_times)
        cpp_avg = sum(cpp_times) / len(cpp_times)

        overhead = py_avg - cpp_avg
        pct_slower = (overhead / cpp_avg) * 100.0

        # Format element size as power of 2
        log2_size = int(math.log2(size))
        size_str = f"2^{log2_size}"

        if py_cpu_times and cpp_cpu_times:
            # Show CPU time comparison for Python interpreter overhead
            py_cpu_avg = sum(py_cpu_times) / len(py_cpu_times)
            cpp_cpu_avg = sum(cpp_cpu_times) / len(cpp_cpu_times)
            cpu_overhead = py_cpu_avg - cpp_cpu_avg

            table_data.append(
                [
                    size_str,
                    format_duration(cpp_avg),
                    format_duration(py_avg),
                    format_percentage(pct_slower / 100.0),
                    format_duration(cpp_cpu_avg),
                    format_duration(py_cpu_avg),
                    format_duration(cpu_overhead),
                ]
            )
        else:
            # Fallback if CPU times not available
            table_data.append(
                [
                    size_str,
                    format_duration(cpp_avg),
                    format_duration(py_avg),
                    format_percentage(pct_slower / 100.0),
                    "N/A",
                    "N/A",
                    "N/A",
                ]
            )

    # Print table using tabulate
    output_fn(tabulate.tabulate(table_data, headers=headers, tablefmt="github"))


def compare_benchmark(py_path, cpp_path, device=None, output_file=None):
    """Compare Python vs C++ benchmark results with hierarchical grouping."""
    if not py_path.exists():
        print(f"Error: Python results not found: {py_path}")
        return False
    if not cpp_path.exists():
        print(f"Error: C++ results not found: {cpp_path}")
        return False

    py_results = utils.read_file(py_path)
    cpp_results = utils.read_file(cpp_path)

    py_measurements = extract_measurements(py_results)
    cpp_measurements = extract_measurements(cpp_results)

    if not py_measurements or not cpp_measurements:
        py_total, py_skipped, py_reasons = summarize_skips(py_results)
        cpp_total, cpp_skipped, cpp_reasons = summarize_skips(cpp_results)
        if not py_measurements:
            print(
                "Python results contain no non-skipped measurements "
                f"({py_skipped}/{py_total} states skipped)."
            )
            if py_reasons:
                print("Python skip reasons (first 3):")
                for r in list(sorted(py_reasons))[:3]:
                    print(f"  - {r.splitlines()[0]}")
        if not cpp_measurements:
            print(
                "C++ results contain no non-skipped measurements "
                f"({cpp_skipped}/{cpp_total} states skipped)."
            )
            if cpp_reasons:
                print("C++ skip reasons (first 3):")
                for r in list(sorted(cpp_reasons))[:3]:
                    print(f"  - {r.splitlines()[0]}")

    # Filter by device if requested
    if device is not None:
        py_measurements = [m for m in py_measurements if m["device"] == device]
        cpp_measurements = [m for m in cpp_measurements if m["device"] == device]

    if not py_measurements or not cpp_measurements:
        print("No matching measurements found!")
        return False

    # Apply standard filters (e.g., OffsetT → I64)
    cpp_measurements = filter_measurements(cpp_measurements, FILTER_AXES)

    # Capture output if writing to file
    output_lines = []

    def output(line=""):
        print(line)
        output_lines.append(line)

    # Header based on file path
    # Get the benchmark name from the path (e.g., "transform/fill" from results/transform/fill_py.json)
    bench_name = py_path.stem.replace("_py", "")
    parent_dir = py_path.parent.name
    if parent_dir != "results":
        bench_name = f"{parent_dir}/{bench_name}"

    output(f"# {bench_name}")
    output()
    output("GPU Time: Mean GPU execution time (cold start, pure kernel)")
    output("  CUDA events (nvbench tag: nv/cold/time/gpu/mean)")
    output("CPU Time: Mean CPU (host) latency")
    output("  Host clock (nvbench tag: nv/cold/time/cpu/mean)")
    output()

    # Get unique devices
    py_devices = sorted(set(m["device"] for m in py_measurements))
    cpp_devices = sorted(set(m["device"] for m in cpp_measurements))

    for device_id in py_devices:
        if device_id not in cpp_devices:
            continue

        # Get device info
        py_device = utils.find_device_by_id(device_id, py_results.get("devices", []))
        device_name = py_device["name"] if py_device else f"Device {device_id}"

        output(f"## [{device_id}] {device_name}")
        output()

        # Filter measurements for this device
        py_device_measurements = [
            m for m in py_measurements if m["device"] == device_id
        ]
        cpp_device_measurements = [
            m for m in cpp_measurements if m["device"] == device_id
        ]

        # Get unique benchmark names (for multi-benchmark files like babelstream)
        py_bench_names = sorted(set(m["benchmark"] for m in py_device_measurements))
        cpp_bench_names = sorted(set(m["benchmark"] for m in cpp_device_measurements))
        common_bench_names = [n for n in py_bench_names if n in cpp_bench_names]

        # Track header depth for markdown formatting
        has_multiple_benchmarks = len(common_bench_names) > 1

        # Process each sub-benchmark
        for bench_name in common_bench_names:
            py_bench = [
                m for m in py_device_measurements if m["benchmark"] == bench_name
            ]
            cpp_bench = [
                m for m in cpp_device_measurements if m["benchmark"] == bench_name
            ]

            if not py_bench or not cpp_bench:
                continue

            # Show benchmark name header if multiple benchmarks
            if has_multiple_benchmarks:
                output(f"### Benchmark: {bench_name}")
                output()

            # Identify grouping axes (algorithm-specific)
            # Use Python measurements as reference (C++ may have extra axes)
            py_grouping_axes = get_grouping_axes(py_bench)
            cpp_grouping_axes = get_grouping_axes(cpp_bench)

            # Only group by axes present in BOTH Python and C++
            common_grouping_axes = [
                a for a in py_grouping_axes if a in cpp_grouping_axes
            ]

            # Get all combinations of grouping axis values
            if common_grouping_axes:
                axis_value_lists = []
                for axis in common_grouping_axes:
                    py_values = set(get_axis_values(py_bench, axis))
                    cpp_values = set(get_axis_values(cpp_bench, axis))
                    common_values = sorted(py_values & cpp_values)
                    axis_value_lists.append(common_values)

                # Generate all combinations
                grouping_combinations = list(product(*axis_value_lists))
            else:
                grouping_combinations = [()]  # Single empty combination

            # Process each grouping combination
            for combo in grouping_combinations:
                # Build filter dict for this combination
                combo_filter = dict(zip(common_grouping_axes, combo))

                # Filter measurements for this combination
                py_combo = filter_by_axes(py_bench, combo_filter)
                cpp_combo = filter_by_axes(cpp_bench, combo_filter)

                if not py_combo or not cpp_combo:
                    continue

                # Show grouping axis values as header if present
                if combo_filter:
                    combo_str = ", ".join(f"{k}={v}" for k, v in combo_filter.items())
                    header_level = "####" if has_multiple_benchmarks else "###"
                    output(f"{header_level} {combo_str}")
                    output()

                # Now group by Type (T axis)
                py_types = get_axis_values(py_combo, "T")
                cpp_types = get_axis_values(cpp_combo, "T")
                common_types = [t for t in py_types if t in cpp_types]

                if common_types:
                    # Multi-type: show separate table per type
                    for type_str in common_types:
                        py_type = filter_by_axes(py_combo, {"T": type_str})
                        cpp_type = filter_by_axes(cpp_combo, {"T": type_str})

                        if not py_type or not cpp_type:
                            continue

                        # Determine header level based on context
                        if combo_filter:
                            type_header_level = (
                                "#####" if has_multiple_benchmarks else "####"
                            )
                        else:
                            type_header_level = (
                                "####" if has_multiple_benchmarks else "###"
                            )

                        output(f"{type_header_level} Type: {type_str}")
                        output()
                        print_comparison_table(py_type, cpp_type, output)
                        output()
                else:
                    # No type axis: show single table
                    print_comparison_table(py_combo, cpp_combo, output)
                    output()

    # Write to file if requested
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("\n".join(output_lines))
        print(f"\nComparison saved to: {output_file}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compare Python vs C++ CCCL benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare by benchmark path
  %(prog)s -b transform/fill
  %(prog)s -b reduce/sum -d 0
  %(prog)s -b scan/exclusive/sum -d 0

  # Compare all supported benchmarks
  %(prog)s

  # Legacy: Compare specific files
  %(prog)s results/transform/fill_py.json results/transform/fill_cpp.json --device 0

Supported benchmarks:
%(benchmarks)s
"""
        % {
            "prog": "python_vs_cpp_summary.py",
            "benchmarks": "\n".join(f"  {b}" for b in SUPPORTED_BENCHMARKS),
        },
    )

    # New interface: benchmark name
    parser.add_argument(
        "-b",
        "--benchmark",
        help="Benchmark path (e.g., transform/fill, reduce/sum). If not specified, compares all supported benchmarks.",
    )
    parser.add_argument(
        "-d", "--device", type=int, help="Filter by GPU device ID (default: all)"
    )
    parser.add_argument("-o", "--output", type=Path, help="Save comparison to file")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Results directory (default: ../results)",
    )

    # Legacy interface: positional args for file paths
    parser.add_argument(
        "python_json", nargs="?", help="Python benchmark results JSON (legacy)"
    )
    parser.add_argument(
        "cpp_json", nargs="?", help="C++ benchmark results JSON (legacy)"
    )

    args = parser.parse_args()

    # Determine mode: legacy (positional args) or new (-b flag)
    if args.python_json and args.cpp_json:
        # Legacy mode: explicit file paths
        py_path = Path(args.python_json)
        cpp_path = Path(args.cpp_json)
        compare_benchmark(py_path, cpp_path, args.device, args.output)

    elif args.benchmark:
        # New mode: single benchmark by path
        if args.benchmark not in SUPPORTED_BENCHMARKS:
            print(
                f"Error: Unknown benchmark '{args.benchmark}'.\n"
                f"Supported benchmarks:\n"
                + "\n".join(f"  {b}" for b in SUPPORTED_BENCHMARKS)
            )
            sys.exit(1)

        py_path, cpp_path, comparison_path = get_result_paths(
            args.results_dir, args.benchmark
        )
        output_path = args.output or comparison_path

        if not compare_benchmark(py_path, cpp_path, args.device, output_path):
            sys.exit(1)

    else:
        # Default: compare all supported benchmarks
        print("Comparing all supported benchmarks...\n")
        any_success = False

        for bench in SUPPORTED_BENCHMARKS:
            py_path, cpp_path, comparison_path = get_result_paths(
                args.results_dir, bench
            )

            if not py_path.exists() or not cpp_path.exists():
                print(f"Skipping {bench}: results not found")
                print(f"  Run: ./run_benchmarks.sh -b {bench}")
                print()
                continue

            print("=" * 72)
            print(f"Benchmark: {bench}")
            print("=" * 72)
            print()

            if compare_benchmark(py_path, cpp_path, args.device, comparison_path):
                any_success = True
            print()

        if not any_success:
            print("No benchmark results found. Run benchmarks first:")
            print("  ./run_benchmarks.sh -b transform/fill")
            sys.exit(1)


if __name__ == "__main__":
    main()
