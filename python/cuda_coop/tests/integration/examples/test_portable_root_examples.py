# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from ...support.paths import PACKAGE_ROOT

_MARKER = "portable-root-sum"
_EXAMPLES = (
    PACKAGE_ROOT / "examples" / "cutlass" / "portable_root_sum.py",
    PACKAGE_ROOT / "examples" / "numba_mlir" / "portable_root_sum.py",
)
_EXPECTED_COOP_CALLS = (
    "coop.this_block",
    "coop.ThreadData",
    "coop.load",
    "coop.store",
    "coop.reduce",
    "block.rank",
)


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return None if parent is None else f"{parent}.{node.attr}"
    return None


def _extract(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = f"# docs: start {_MARKER}"
    end = f"# docs: end {_MARKER}"
    assert [line.strip() for line in lines].count(start) == 1
    assert [line.strip() for line in lines].count(end) == 1
    start_index = next(i for i, line in enumerate(lines) if line.strip() == start)
    end_index = next(i for i, line in enumerate(lines) if line.strip() == end)
    assert start_index < end_index
    return textwrap.dedent("\n".join(lines[start_index + 1 : end_index]))


def _kernel(tree: ast.Module, path: Path) -> ast.FunctionDef:
    kernels = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "portable_root_sum_kernel"
    ]
    assert len(kernels) == 1, path
    return kernels[0]


def test_portable_root_examples_have_matching_extractable_kernel_calls():
    for path in _EXAMPLES:
        tree = ast.parse(_extract(path), filename=str(path))
        root_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "cuda"
            and [(alias.name, alias.asname) for alias in node.names] == [("coop", None)]
        ]
        assert len(root_imports) == 1, path

        kernel = _kernel(tree, path)
        calls = sorted(
            (node.lineno, node.col_offset, _dotted_name(node.func))
            for node in ast.walk(kernel)
            if isinstance(node, ast.Call)
            and (
                (_dotted_name(node.func) or "").startswith("coop.")
                or _dotted_name(node.func) == "block.rank"
            )
        )
        assert tuple(name for _, _, name in calls) == _EXPECTED_COOP_CALLS, path

        qualified = sorted(
            (node.lineno, name)
            for node in ast.walk(kernel)
            if isinstance(node, ast.Attribute)
            and (name := _dotted_name(node)) is not None
            and name.startswith("cuda.coop")
        )
        assert qualified == [], path


def test_portable_root_examples_expose_runtime_validation_entrypoints():
    for path in _EXAMPLES:
        package_tree = ast.parse(
            (path.parent / "__init__.py").read_text(encoding="utf-8")
        )
        exports = next(
            ast.literal_eval(node.value)
            for node in package_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        )
        assert path.stem in exports

        module_tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = {
            node.name
            for node in ast.walk(module_tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert {"run_example", "main"} <= functions
