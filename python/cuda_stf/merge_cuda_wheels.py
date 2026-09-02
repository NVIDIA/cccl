#!/usr/bin/env python3
"""
Script to merge CUDA-specific cuda-stf wheels into a single multi-CUDA wheel.

This script takes wheels built for different CUDA versions (cu12, cu13) and merges them
into a single wheel that supports both CUDA versions.

Each wheel contains a CUDA-specific build in a versioned directory:
- `cuda/stf/_experimental/cu<version>` -- cccl.c.experimental.stf and
  cuda.stf._experimental bindings (Linux only)

This script merges those directories so the final wheel supports both CUDA versions.
At runtime, the shim module chooses the right extension from the detected CUDA version
(see `cuda/stf/_experimental/_stf_bindings.py`).
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

_CUDA_WHEEL_SUFFIX_RE = re.compile(r"\.cu(?P<version>\d+)(?=\.whl$)")


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


def cuda_version_from_wheel_name(wheel_name: str) -> str:
    match = _CUDA_WHEEL_SUFFIX_RE.search(wheel_name)
    if match is None:
        raise ValueError(f"Could not find CUDA suffix in wheel name: {wheel_name}")
    return match.group("version")


def strip_cuda_suffix(wheel_name: str) -> str:
    return _CUDA_WHEEL_SUFFIX_RE.sub("", wheel_name)


# Every input wheel must ship exactly this per-CUDA-major subtree (relative to
# the wheel root, under its ``cu<major>`` directory).
_VERSION_SUBDIRS = [
    Path("cuda") / "stf" / "_experimental",
]


def _cuda_subtree_dirs(wheel_dir: Path, cuda_version: str) -> List[Path]:
    """Absolute paths of the ``cu<version>`` subtrees expected in *wheel_dir*."""
    return [wheel_dir / parent / f"cu{cuda_version}" for parent in _VERSION_SUBDIRS]


def _require_cuda_subtrees(wheel_dir: Path, cuda_version: str, wheel_name: str) -> None:
    """Fail unless *wheel_dir* contains every expected non-empty CUDA subtree."""
    for subtree in _cuda_subtree_dirs(wheel_dir, cuda_version):
        if not subtree.is_dir():
            raise RuntimeError(
                f"wheel {wheel_name!r} is missing its expected CUDA subtree "
                f"{subtree.relative_to(wheel_dir)}"
            )
        if not any(subtree.iterdir()):
            raise RuntimeError(
                f"wheel {wheel_name!r} has an empty CUDA subtree "
                f"{subtree.relative_to(wheel_dir)}"
            )


def merge_wheels(wheels: List[Path], output_dir: Path) -> Path:
    """Merge multiple wheels into a single wheel with version-specific binaries."""
    print("\n=== Merging wheels ===")
    print(f"Input wheels: {[w.name for w in wheels]}")

    # Reject duplicate CUDA majors up front: merging two cu12 wheels (say) would
    # otherwise silently clobber or collide on the same cu12 subtree.
    versions = [cuda_version_from_wheel_name(w.name) for w in wheels]
    seen = set()
    for version, wheel in zip(versions, wheels):
        if version in seen:
            raise RuntimeError(
                f"duplicate CUDA major cu{version} among input wheels "
                f"(offending wheel: {wheel.name})"
            )
        seen.add(version)

    if len(wheels) == 1:
        # Single wheel, just copy it and remove CUDA version suffix
        output_dir.mkdir(parents=True, exist_ok=True)
        final_wheel = output_dir / strip_cuda_suffix(wheels[0].name)
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
                if item.is_dir() and item.name.startswith("cuda_stf"):
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

        # Use the first wheel as the base and merge binaries from others.
        base_wheel = extracted_wheels[0]

        # Every input wheel (including the base) must actually contain its own
        # CUDA subtree; otherwise the merged wheel would be missing a backend.
        for i, wheel_dir in enumerate(extracted_wheels):
            _require_cuda_subtrees(wheel_dir, versions[i], wheels[i].name)

        # Copy the version-specific directories from the other wheels into the
        # base wheel, refusing to overwrite anything already present.
        for i, wheel_dir in enumerate(extracted_wheels):
            cuda_version = versions[i]
            if i == 0:
                # For base wheel, do nothing (its own subtree stays in place).
                continue
            for parent in _VERSION_SUBDIRS:
                version_dir = parent / f"cu{cuda_version}"
                src = wheel_dir / version_dir
                dst = base_wheel / version_dir
                if dst.exists():
                    raise RuntimeError(
                        f"refusing to merge: {version_dir} already exists in the base "
                        f"wheel (conflicting content from {wheels[i].name})"
                    )
                print(f"  Copying {version_dir} to {base_wheel}")
                shutil.copytree(src, dst)

        # Repack the merged wheel
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a clean wheel name without CUDA version suffixes
        base_wheel_name = strip_cuda_suffix(wheels[0].name)

        # Snapshot existing wheels so we can unambiguously identify the one
        # produced by ``wheel pack`` (its exact name is derived from metadata
        # and may not match base_wheel_name byte-for-byte).
        wheels_before = set(output_dir.glob("*.whl"))

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

        # Identify exactly the wheel that ``wheel pack`` just produced.
        new_wheels = sorted(set(output_dir.glob("*.whl")) - wheels_before)
        if len(new_wheels) != 1:
            raise RuntimeError(
                "expected exactly one new wheel from 'wheel pack', found "
                f"{[w.name for w in new_wheels]}"
            )
        merged_wheel = new_wheels[0]
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

    print("CUDA STF Wheel Merger")
    print("=====================")

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
        run_command([sys.executable, "-m", "wheel", "--help"])
    except Exception:
        print("Error: wheel package not available. Install with: pip install wheel")
        sys.exit(1)

    # Merge the wheels
    merged_wheel = merge_wheels(wheels, output_dir)
    print(f"\nMerge complete! Output: {merged_wheel}")


if __name__ == "__main__":
    main()
