# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import textwrap

import pytest

from ._imports import (
    SOURCE_ROOT,
    assert_modules_import_from_source,
    run_python_with_source,
)


@pytest.mark.parametrize(
    "module_name",
    ("cuda.coop._core", "cuda.coop._core.warp"),
)
def test_backend_neutral_modules_import_from_source_tree(module_name):
    assert_modules_import_from_source(module_name)


def test_cuda_coop_namespace_imports_from_cuda_coop_source():
    script = textwrap.dedent(
        f"""
        import importlib.util
        from pathlib import Path

        import cuda.coop

        source_root = Path({str(SOURCE_ROOT)!r}).resolve()
        coop_path = source_root / "cuda" / "coop"
        coop_paths = {{Path(path).resolve() for path in cuda.coop.__path__}}
        assert coop_path in coop_paths, (coop_path, coop_paths)
        assert importlib.util.find_spec("cuda.coop.numba_mlir") is not None
        """
    )
    result = run_python_with_source(script)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source_first", (True, False))
def test_cuda_coop_package_extends_path_for_provider_coinstallation(
    source_first,
    tmp_path,
):
    provider_root = tmp_path / "provider-package"
    provider_coop = provider_root / "cuda" / "coop"
    provider_coop.mkdir(parents=True)
    (provider_coop / "future_backend.py").write_text(
        '"""Synthetic independently installed provider."""\n',
        encoding="utf-8",
    )
    path_roots = (
        (SOURCE_ROOT, provider_root) if source_first else (provider_root, SOURCE_ROOT)
    )
    script = textwrap.dedent(
        f"""
        import importlib.util
        from pathlib import Path

        import cuda.coop

        coop_paths = {{Path(path).resolve() for path in cuda.coop.__path__}}
        expected_paths = {{
            Path(root).resolve() / "cuda" / "coop"
            for root in {[str(root) for root in path_roots]!r}
        }}
        assert expected_paths.issubset(coop_paths), (expected_paths, coop_paths)
        assert isinstance(cuda.coop.__version__, str)
        assert cuda.coop.__version__
        assert importlib.util.find_spec("cuda.coop.future_backend") is not None
        assert importlib.util.find_spec("cuda.coop.numba_mlir") is not None
        """
    )
    result = run_python_with_source(
        script,
        roots=path_roots,
        inherit_pythonpath=False,
    )
    assert result.returncode == 0, result.stderr


def test_retired_numba_namespaces_are_not_shipped():
    assert not tuple((SOURCE_ROOT / "cuda" / "coop" / "numba").rglob("*.py"))
    assert not tuple((SOURCE_ROOT / "cuda" / "coop" / "_experimental").rglob("*.py"))
    pyproject = (SOURCE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"cuda/coop/numba"' not in pyproject
