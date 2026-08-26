# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_CONSUMER_ROOT = Path(__file__).with_name("typing")
_VALID_CONSUMERS = (
    "portable_consumer.py",
    "cutlass_consumer.py",
    "numba_consumer.py",
)


def _mypy_args(mypy: str, cache_dir: Path) -> list[str]:
    return [
        mypy,
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
    env = os.environ.copy()
    env["MYPYPATH"] = str(stub_root)
    return subprocess.run(
        [*mypy_args, *(str(consumer) for consumer in consumers)],
        cwd=stub_root.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_installed_stubs_pass_strict_consumer_type_check(tmp_path: Path) -> None:
    mypy = shutil.which("mypy")
    if mypy is None:
        pytest.skip("mypy is not installed")

    spec = importlib.util.find_spec("cuda.coop")
    assert spec is not None
    assert spec.submodule_search_locations is not None
    package_root = Path(next(iter(spec.submodule_search_locations)))

    stub_root = tmp_path / "stubs" / "cuda" / "coop"
    for source in package_root.rglob("*.pyi"):
        destination = stub_root / source.relative_to(package_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(package_root / "py.typed", stub_root / "py.typed")

    consumer_root = tmp_path / "consumers"
    consumer_root.mkdir()
    valid_consumers = []
    for name in _VALID_CONSUMERS:
        destination = consumer_root / name
        shutil.copyfile(_CONSUMER_ROOT / name, destination)
        valid_consumers.append(destination)

    mypy_args = _mypy_args(mypy, tmp_path / "mypy-cache")
    result = _run_mypy(
        mypy_args,
        valid_consumers,
        stub_root=tmp_path / "stubs",
    )
    assert result.returncode == 0, result.stdout + result.stderr

    invalid_consumer = tmp_path / "invalid_consumer.py"
    shutil.copyfile(_CONSUMER_ROOT / "invalid_consumer.py", invalid_consumer)
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
            r"invalid_consumer\.py:(\d+): error: .* (\[[a-z-]+\])$",
            invalid_output,
            flags=re.MULTILINE,
        )
    }
    assert diagnostics == {
        (16, "[assert-type]"),
        (22, "[assert-type]"),
        (32, "[arg-type]"),
        (36, "[assignment]"),
    }
