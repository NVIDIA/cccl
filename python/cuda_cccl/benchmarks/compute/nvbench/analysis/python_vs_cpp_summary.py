#!/usr/bin/env python

"""
Compare Python cuda.compute vs C++ CCCL performance.
Shows overhead and percentage differences.
"""

import argparse
import math
import sys
from pathlib import Path

import tabulate
import utils


def extract_measurements(results):
    """Extract all state measurements with GPU and CPU time."""
    measurements = []

    for benchmark in results.get("benchmarks", []):
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
                        "device": state["device"],
                        "axes": axes,
                        "gpu_time": gpu_time,
                        "cpu_time": cpu_time,  # May be None
                    }
                )

    return measurements


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


def main():
    parser = argparse.ArgumentParser(
        description="Compare Python vs C++ CCCL benchmark results"
    )
    parser.add_argument("python_json", help="Python benchmark results (JSON)")
    parser.add_argument("cpp_json", help="C++ benchmark results (JSON)")
    parser.add_argument(
        "--device",
        type=int,
        help="Only compare specific device ID (default: all devices)",
    )
    args = parser.parse_args()

    py_path = Path(args.python_json)
    cpp_path = Path(args.cpp_json)

    if not py_path.exists():
        print(f"Error: Python results not found: {py_path}")
        sys.exit(1)
    if not cpp_path.exists():
        print(f"Error: C++ results not found: {cpp_path}")
        sys.exit(1)

    py_results = utils.read_file(py_path)
    cpp_results = utils.read_file(cpp_path)

    py_measurements = extract_measurements(py_results)
    cpp_measurements = extract_measurements(cpp_results)

    # Filter by device if requested
    if args.device is not None:
        py_measurements = [m for m in py_measurements if m["device"] == args.device]
        cpp_measurements = [m for m in cpp_measurements if m["device"] == args.device]

    if not py_measurements or not cpp_measurements:
        print("No matching measurements found!")
        sys.exit(1)

    # Get benchmark name
    bench_name = py_results.get("benchmarks", [{}])[0].get("name", "unknown")

    print(f"# {bench_name}\n")
    print("GPU Time: Mean GPU execution time (cold start, pure kernel)")
    print("  CUDA events (nvbench tag: nv/cold/time/gpu/mean)")
    print("CPU Time: Mean CPU (host) latency")
    print("  Host clock (nvbench tag: nv/cold/time/cpu/mean)")
    print()

    # Get unique devices
    py_devices = sorted(set(m["device"] for m in py_measurements))
    cpp_devices = sorted(set(m["device"] for m in cpp_measurements))

    for device_id in py_devices:
        if device_id not in cpp_devices:
            continue

        # Get device info
        py_device = utils.find_device_by_id(device_id, py_results.get("devices", []))
        device_name = py_device["name"] if py_device else f"Device {device_id}"

        print(f"## [{device_id}] {device_name}\n")

        # Filter measurements for this device
        py_device_measurements = [
            m for m in py_measurements if m["device"] == device_id
        ]
        cpp_device_measurements = [
            m for m in cpp_measurements if m["device"] == device_id
        ]

        # Get unique types (if present)
        types = sorted(set(m["axes"].get("T", "") for m in py_device_measurements))
        has_types = any(types)

        # Group by type (if multi-type benchmark) or just element size
        if has_types and len(types) > 1:
            # Multi-type benchmark: show separate table per type
            for type_str in types:
                if not type_str:
                    continue

                print(f"### Type: {type_str}\n")

                py_type_measurements = [
                    m for m in py_device_measurements if m["axes"].get("T") == type_str
                ]
                cpp_type_measurements = [
                    m for m in cpp_device_measurements if m["axes"].get("T") == type_str
                ]

                _print_comparison_table(py_type_measurements, cpp_type_measurements)
                print()
        else:
            # Single-type or no type axis: show one table
            _print_comparison_table(py_device_measurements, cpp_device_measurements)
            print()


def _print_comparison_table(py_measurements, cpp_measurements):
    """Print comparison table for given measurements."""
    # Group by element size
    element_sizes = sorted(
        set(int(m["axes"].get("Elements", 0)) for m in py_measurements)
    )

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

        # Average if multiple measurements
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
    print(tabulate.tabulate(table_data, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
