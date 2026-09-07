# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-isolated subprocess helpers for public import contracts."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from ...support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT


def run_python_with_source(
    script: str,
    *,
    roots: Sequence[Path] = (SOURCE_ROOT,),
    inherit_pythonpath: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run Python without site packages, rooted at the requested source trees."""
    env = os.environ.copy()
    paths = [os.fspath(root) for root in roots]
    inherited = env.get("PYTHONPATH") if inherit_pythonpath else None
    if inherited:
        paths.append(inherited)
    env["PYTHONPATH"] = os.pathsep.join(paths)

    return subprocess.run(
        [sys.executable, "-S", "-B", "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def optional_dependencies() -> dict[str, list[str]]:
    """Load cuda.coop extras without importing any optional backend."""
    with (SOURCE_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]["optional-dependencies"]


def assert_modules_import_from_source(*module_names: str) -> None:
    """Assert that lazy backend modules resolve from this checkout."""
    script = textwrap.dedent(
        f"""
        import importlib
        from pathlib import Path

        source_root = Path({str(SOURCE_ROOT)!r}).resolve()
        for module_name in {module_names!r}:
            module = importlib.import_module(module_name)
            module_file = Path(module.__file__).resolve()
            assert module.__name__ == module_name
            assert module_file.is_relative_to(source_root), (
                f"{{module.__name__}} resolved to {{module_file}}, "
                f"expected path under {{source_root}}"
            )
        """
    )
    result = run_python_with_source(script)
    assert result.returncode == 0, result.stderr
