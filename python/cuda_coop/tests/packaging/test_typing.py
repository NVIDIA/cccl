# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).parents[2]
_CONSUMER_ROOT = Path(__file__).with_name("typing")
_VALID_CONSUMERS = ("common_consumer.py", "numba_consumer.py")
_THREAD_GROUP_HIERARCHY_METHODS = frozenset(
    {
        "count",
        "count_as",
        "is_member",
        "rank",
        "rank_as",
        "sync",
        "sync_aligned",
    }
)


def _package_stub_source() -> Path:
    try:
        distribution = importlib.metadata.distribution("cuda-coop")
    except importlib.metadata.PackageNotFoundError:
        return _PACKAGE_ROOT / "cuda" / "coop"

    installed = Path(distribution.locate_file("cuda/coop"))
    assert installed.is_dir(), f"installed cuda-coop package is missing: {installed}"
    return installed


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
    return {
        (line_number, error_code)
        for line_number, line in enumerate(
            consumer.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if (match := re.search(r"# expected-error: (\[[a-z-]+\])", line))
        for error_code in (match.group(1),)
    }


def _literal_exports(path: Path) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    declaration = next(
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    return {
        ast.literal_eval(item)
        for item in declaration.elts
        if not isinstance(item, ast.Starred)
    }


@pytest.mark.parametrize(
    "relative_path",
    ("__init__.pyi", "_core/api/__init__.pyi", "numba_mlir/__init__.pyi"),
    ids=("root", "core", "qualified"),
)
def test_public_stub_exports_match_runtime(relative_path: str) -> None:
    package_root = _package_stub_source()
    stub = package_root / relative_path
    runtime_exports = _literal_exports(stub.with_suffix(".py"))
    if relative_path == "__init__.pyi":
        # The root expands the common API exports without importing a compiler.
        runtime_exports |= _literal_exports(package_root / "_core/api/__init__.py")

    assert _literal_exports(stub) == runtime_exports


@pytest.mark.parametrize(
    "relative_path",
    ("_core/api/thread_group.pyi", "numba_mlir/_thread_group.pyi"),
    ids=("common", "qualified"),
)
def test_thread_group_stubs_expose_hierarchy_operations(relative_path: str) -> None:
    stub = _package_stub_source() / relative_path
    module = ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub))
    thread_group = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ThreadGroup"
    )
    methods = {
        node.name for node in thread_group.body if isinstance(node, ast.FunctionDef)
    }

    assert _THREAD_GROUP_HIERARCHY_METHODS <= methods


def test_public_stubs_pass_strict_consumer_type_checks(tmp_path: Path) -> None:
    if importlib.util.find_spec("mypy") is None:
        pytest.skip("mypy is not installed")

    package_root = _package_stub_source()
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

    mypy_args = _mypy_args(tmp_path / "mypy-cache")
    result = _run_mypy(
        mypy_args,
        valid_consumers,
        stub_root=tmp_path / "stubs",
    )
    assert result.returncode == 0, result.stdout + result.stderr

    invalid_consumer = consumer_root / "invalid_consumer.py"
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
    assert diagnostics == _expected_diagnostics(invalid_consumer), invalid_output
