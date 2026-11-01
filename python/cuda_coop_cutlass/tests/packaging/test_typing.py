# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check public stubs through isolated valid and invalid consumer programs.

Copy only the imported package's ``.pyi`` stubs and ``py.typed`` marker to a
temporary root. Implementation modules and a separately installed
cuda-coop-cutlass cannot supply missing public declarations.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import cuda.coop.cutlass as coop

_CONSUMER_ROOT = Path(__file__).with_name("typing")


def _mypy_args(cache_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mypy",
        "--strict",
        "--python-version",
        f"{sys.version_info.major}.{sys.version_info.minor}",
        "--disallow-any-unimported",
        "--show-error-codes",
        "--cache-dir",
        str(cache_dir),
    ]


def _run_mypy(
    mypy_args: list[str],
    consumers: list[Path],
    *,
    stub_root: Path,
) -> subprocess.CompletedProcess[str]:
    """Type-check copied consumers against only the selected stub root."""
    environment = os.environ.copy()
    environment["MYPYPATH"] = str(stub_root)
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [*mypy_args, *(str(consumer) for consumer in consumers)],
        cwd=stub_root.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _expected_diagnostics(consumer: Path) -> set[tuple[int, str]]:
    """Read expected error codes at their current consumer source lines."""
    return {
        (line_number, error_code)
        for line_number, line in enumerate(
            consumer.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if (match := re.search(r"# expected-error: (\[[a-z-]+\])", line))
        for error_code in (match.group(1),)
    }


@pytest.mark.parametrize(
    "consumer_family",
    sorted(
        path.stem.removesuffix("_consumer")
        for path in _CONSUMER_ROOT.glob("*_consumer.py")
        if not path.name.startswith("invalid_")
    ),
)
def test_public_stubs_pass_strict_consumer_type_checks(
    tmp_path: Path, consumer_family: str
) -> None:
    """Check accepted calls and exact diagnostics against isolated stubs."""

    if importlib.util.find_spec("mypy") is None:
        pytest.fail("mypy is required for the public typing gate")
    if importlib.util.find_spec("cutlass") is None:
        pytest.fail("CUTLASS DSL is required for public scalar typing")

    package_root = Path(coop.__file__).parent
    stub_root = tmp_path / "stubs" / "cuda" / "coop" / "cutlass"
    for source in package_root.rglob("*.pyi"):
        destination = stub_root / source.relative_to(package_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(package_root / "py.typed", stub_root / "py.typed")
    # Make both namespace parents resolve only the copied standalone stubs.
    (stub_root.parent / "__init__.pyi").touch()
    (stub_root.parent.parent / "__init__.pyi").touch()

    consumer_root = tmp_path / "consumers"
    consumer_root.mkdir()
    valid_consumers = []
    consumer_names = (f"{consumer_family}_consumer.py",)
    for name in consumer_names:
        destination = consumer_root / name
        shutil.copyfile(_CONSUMER_ROOT / name, destination)
        valid_consumers.append(destination)

    mypy_args = _mypy_args(tmp_path / "mypy-cache")
    mypy_args.extend(
        ["--config-file", str(_CONSUMER_ROOT / "cutlass_mypy.ini")]
    )
    result = _run_mypy(
        mypy_args,
        valid_consumers,
        stub_root=tmp_path / "stubs",
    )
    assert result.returncode == 0, result.stdout + result.stderr

    invalid_name = f"invalid_{consumer_family}_consumer.py"
    invalid_consumer = consumer_root / invalid_name
    shutil.copyfile(_CONSUMER_ROOT / invalid_name, invalid_consumer)
    invalid_result = _run_mypy(
        mypy_args,
        [invalid_consumer],
        stub_root=tmp_path / "stubs",
    )
    invalid_output = invalid_result.stdout + invalid_result.stderr
    assert invalid_result.returncode == 1, invalid_output

    diagnostics = {
        (int(line), code)
        for line, code in re.findall(
            rf"{re.escape(invalid_name)}:(\d+): error: .* (\[[a-z-]+\])$",
            invalid_output,
            flags=re.MULTILINE,
        )
    }
    assert diagnostics == _expected_diagnostics(invalid_consumer), (
        invalid_output
    )
