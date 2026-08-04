#!/usr/bin/env python3
"""
Script to merge CUDA-specific wheels into a single multi-CUDA wheel.

This script takes wheels built for different CUDA versions (cu12, cu13) and merges them
into a single wheel that supports both CUDA versions.

In particular, each wheel contains a CUDA-specific build of the `cccl.c.parallel` library
and the associated bindings. These are present in the directory `compute/cu<version>`.
For example, for a wheel built with CUDA 12, the directory is `compute/cu12`,
and for a wheel built with CUDA 13, the directory is `compute/cu13`.
This script merges these directories into a single wheel that supports both CUDA versions, i.e.,
containing both `compute/cu12` and `compute/cu13`.
At runtime, a shim module `compute/_bindings.py` is used to import the appropriate
CUDA-specific bindings. See `compute/_bindings.py` for more details.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command with error handling."""
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"  Working directory: {cwd}")

    # Callers construct argv lists directly; no shell parses wheel paths.
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)

    return result


def _is_cuda_major_payload(path: Path) -> bool:
    parts = path.parts
    if parts[0:2] != ("cuda", "compute"):
        return False
    try:
        cuda_directory = parts[2]
    except IndexError:
        return False
    return cuda_directory.startswith("cu") and cuda_directory[2:].isdigit()


def _shared_logical_contents(root: Path) -> dict[Path, bytes]:
    """Return payload that must be identical in every CUDA-major wheel."""
    contents: dict[Path, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _is_cuda_major_payload(relative):
            continue
        # wheel pack regenerates RECORD after the CUDA-major trees are merged.
        if relative.name == "RECORD" and relative.parent.name.endswith(".dist-info"):
            continue
        contents[relative] = path.read_bytes()
    return contents


def _validate_shared_contents(extracted_wheels: list[Path], wheels: list[Path]) -> None:
    base_contents = _shared_logical_contents(extracted_wheels[0])
    for wheel, extracted in zip(wheels[1:], extracted_wheels[1:], strict=True):
        contents = _shared_logical_contents(extracted)
        if contents == base_contents:
            continue

        missing = sorted(str(path) for path in base_contents.keys() - contents.keys())
        extra = sorted(str(path) for path in contents.keys() - base_contents.keys())
        changed = sorted(
            str(path)
            for path in base_contents.keys() & contents.keys()
            if base_contents[path] != contents[path]
        )
        raise RuntimeError(
            f"Cannot merge {wheel.name}: non-CUDA-major wheel contents differ "
            f"(missing={missing}, extra={extra}, changed={changed})"
        )


def merge_wheels(wheels: list[Path], output_dir: Path) -> Path:
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
                    sys.executable,
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
                if item.is_dir() and item.name.startswith("cuda_compute"):
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

        # Python sources, HostJIT resources, build metadata, and dist-info must
        # be identical across CUDA-major builds. Only cuda/compute/cuXX may
        # differ; silently taking the first copy would hide a divergent build.
        _validate_shared_contents(extracted_wheels, wheels)

        # now copy the version-specific directory from other wheels
        # into the appropriate place in the base wheel
        for i, wheel_dir in enumerate(extracted_wheels):
            cuda_version = wheels[i].name.split(".cu")[1].split(".")[0]
            if i == 0:
                # For base wheel, do nothing
                continue
            else:
                version_dir = Path("cuda") / "compute" / f"cu{cuda_version}"
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
                sys.executable,
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


def main() -> None:
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

    print("cuda-compute Wheel Merger")
    print("=========================")

    # Convert wheel paths to Path objects and validate
    wheels: list[Path] = []
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

    # Check that we have the wheel tool available. run_command already reports
    # the module error and exits cleanly when it is unavailable.
    run_command([sys.executable, "-m", "wheel", "--help"])

    # Merge the wheels
    merged_wheel = merge_wheels(wheels, output_dir)
    print(f"\nMerge complete! Output: {merged_wheel}")


if __name__ == "__main__":
    main()
