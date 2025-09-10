#!/usr/bin/env python3
"""
Script to merge CUDA-specific wheels into a single multi-CUDA wheel.

This script takes wheels built for different CUDA versions (cu12, cu13) and merges them
into a single wheel that supports both CUDA versions.

In particular, each wheel contains a CUDA-specific build of the `cccl.c.parallel` library
and the associated bindings. These are present in the directory `parallel/experimental/cu<version>`.
For example, for a wheel built with CUDA 12, the directory is `parallel/experimental/cu12`,
and for a wheel built with CUDA 13, the directory is `parallel/experimental/cu13`.
This script merges these directories into a single wheel that supports both CUDA versions, i.e.,
containing both `parallel/experimental/cu12` and `parallel/experimental/cu13`.
At runtime, a shim module `parallel/experimental/_bindings.py` is used to import the appropriate
CUDA-specific bindings. See `parallel/experimental/_bindings.py` for more details.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List


def run_command(
    cmd: List[str], cwd: Path = None, env: dict = None
) -> subprocess.CompletedProcess:
    """Run a command with error handling."""
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"  Working directory: {cwd}")

    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)

    return result


def merge_wheels(wheels: List[Path], output_dir: Path) -> Path:
    """Merge multiple wheels into a single wheel with version-specific binaries."""
    print("\n=== Merging wheels ===")
    print(f"Input wheels: {[w.name for w in wheels]}")

    if len(wheels) == 1:
        # Single wheel, just copy it and remove CUDA version suffix
        output_dir.mkdir(parents=True, exist_ok=True)
        final_wheel = output_dir / wheels[0].name.replace(
            f".cu{wheels[0].name.split('.cu')[1].split('.')[0]}.whl", ".whl"
        )
        shutil.copy2(wheels[0], final_wheel)
        print(f"Single wheel copied to: {final_wheel}")
        return final_wheel

    # Extract all wheels to temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        extracted_wheels = []

        for i, wheel in enumerate(wheels):
            print(f"Extracting wheel {i + 1}/{len(wheels)}: {wheel.name}")
            # Extract wheel - wheel unpack creates the directory itself
            run_command(
                [
                    "python",
                    "-m",
                    "wheel",
                    "unpack",
                    str(wheel),
                    "--dest",
                    str(temp_path),
                ]
            )

            # Find the extracted directory (wheel unpack creates a subdirectory)
            extract_dir = None
            for item in temp_path.iterdir():
                if item.is_dir() and item.name.startswith("cuda_cccl"):
                    extract_dir = item
                    break

            if not extract_dir:
                raise RuntimeError(
                    f"Could not find extracted wheel directory for {wheel.name}"
                )

            # Rename to our expected name
            expected_name = temp_path / f"wheel_{i}"
            extract_dir.rename(expected_name)
            extract_dir = expected_name

            extracted_wheels.append(extract_dir)

        # Use the first wheel as the base and merge binaries from others
        base_wheel = extracted_wheels[0]

        # now copy the version-specific directory from other wheels
        # into the appropriate place in the base wheel
        for i, wheel_dir in enumerate(extracted_wheels):
            cuda_version = wheels[i].name.split(".cu")[1].split(".")[0]
            if i == 0:
                # For base wheel, do nothing
                continue
            else:
                version_dir = (
                    Path("cuda")
                    / "cccl"
                    / "parallel"
                    / "experimental"
                    / f"cu{cuda_version}"
                )
                # Copy from other wheels
                print(f"  Copying {version_dir} to {base_wheel}")
                shutil.copytree(wheel_dir / version_dir, base_wheel / version_dir)

        # Repack the merged wheel
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a clean wheel name without CUDA version suffixes
        base_wheel_name = wheels[0].name
        # Remove any .cu* suffix from the wheel name
        if ".cu" in base_wheel_name:
            base_wheel_name = base_wheel_name.split(".cu")[0] + ".whl"

        print(f"Repacking merged wheel as: {base_wheel_name}")
        run_command(
            [
                "python",
                "-m",
                "wheel",
                "pack",
                str(base_wheel),
                "--dest-dir",
                str(output_dir),
            ]
        )

        # Find the output wheel
        output_wheels = list(output_dir.glob("*.whl"))
        if not output_wheels:
            raise RuntimeError("Failed to create merged wheel")

        merged_wheel = output_wheels[0]
        print(f"Successfully merged wheel: {merged_wheel}")
        return merged_wheel


def main():
    """Main merge script."""
    parser = argparse.ArgumentParser(
        description="Merge CUDA-specific wheels into a single multi-CUDA wheel"
    )
    parser.add_argument(
        "wheels", nargs="+", help="Paths to the CUDA-specific wheels to merge"
    )
    parser.add_argument(
        "--output-dir", "-o", default="dist", help="Output directory for merged wheel"
    )

    args = parser.parse_args()

    print("CUDA CCCL Wheel Merger")
    print("======================")

    # Convert wheel paths to Path objects and validate
    wheels = []
    for wheel_path in args.wheels:
        wheel = Path(wheel_path)
        if not wheel.exists():
            print(f"Error: Wheel not found: {wheel}")
            sys.exit(1)
        if not wheel.name.endswith(".whl"):
            print(f"Error: Not a wheel file: {wheel}")
            sys.exit(1)
        wheels.append(wheel)

    if not wheels:
        print("Error: No wheels provided")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # Check that we have wheel tool available
    try:
        run_command(["python", "-m", "wheel", "--help"])
    except Exception:
        print("Error: wheel package not available. Install with: pip install wheel")
        sys.exit(1)

    # Merge the wheels
    merged_wheel = merge_wheels(wheels, output_dir)
    print(f"\nMerge complete! Output: {merged_wheel}")


if __name__ == "__main__":
    main()
