# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Keep documented signatures and source excerpts aligned with the API."""

from __future__ import annotations

import ast
import copy
import re
from pathlib import Path

from ...support.paths import PACKAGE_ROOT, REPO_ROOT

DOCS_ROOT = REPO_ROOT / "docs"
TILE_DOCS = DOCS_ROOT / "tile"
TILE_SOURCE = PACKAGE_ROOT / "cuda" / "coop" / "tile"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(_read(path), filename=str(path)).body
        if isinstance(node, ast.FunctionDef)
    }


def _exports() -> set[str]:
    tree = ast.parse(_read(TILE_SOURCE / "__init__.py"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_API_SYMBOLS"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError("Tile primitive export list not found")


def _parameters(function: ast.FunctionDef) -> str:
    arguments = copy.deepcopy(function.args)
    for node in ast.walk(arguments):
        if isinstance(node, ast.arg):
            node.annotation = None
    return ast.dump(arguments, include_attributes=False)


def _directives(page: Path, name: str):
    pattern = re.compile(rf"^( *)\.\. {re.escape(name)}::(.*)$")
    lines = _read(page).splitlines(keepends=True)
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match is None:
            continue
        indent = len(match.group(1))
        body = []
        for following in lines[index + 1 :]:
            if (
                following.strip()
                and len(following) - len(following.lstrip()) <= indent
            ):
                break
            body.append(following[indent:])
        yield match.group(2), "".join(body)


def _resolve(page: Path, target: str) -> Path:
    if target.startswith("/"):
        return (DOCS_ROOT / target.lstrip("/")).resolve()
    return (page.parent / target).resolve()


def _literal_sources():
    for page in TILE_DOCS.rglob("*.rst"):
        for target, body in _directives(page, "literalinclude"):
            options = dict(
                re.findall(r"^\s+:([\w-]+): ?(.*)$", body, re.MULTILINE)
            )
            yield page, _resolve(page, target.strip()), options


def test_tile_sphinx_reference_matches_public_signatures():
    documented = {}
    for page in (TILE_DOCS / "python" / "primitives").glob("*.rst"):
        for signature, _ in _directives(page, "py:function"):
            function = ast.parse(f"def {signature.strip()}:\n    ...\n").body[0]
            assert function.name not in documented, function.name
            documented[function.name] = _parameters(function)

    exports = _exports()
    assert documented.keys() == exports
    for path in (TILE_SOURCE / "_api.py", TILE_SOURCE / "__init__.pyi"):
        functions = _functions(path)
        for name in exports:
            assert documented[name] == _parameters(functions[name]), (
                path,
                name,
            )


def test_tile_sphinx_literal_examples_select_valid_source():
    sources = list(_literal_sources())
    assert sources
    for page, path, options in sources:
        lines = _read(path).splitlines(keepends=True)
        for option in ("start-at", "start-after", "end-at", "end-before"):
            if option not in options:
                continue
            matching = [
                index
                for index, line in enumerate(lines)
                if options[option] in line
            ]
            assert matching, (page, path, option, options[option])
            index = matching[0]
            if option == "start-at":
                lines = lines[index:]
            elif option == "start-after":
                lines = lines[index + 1 :]
            elif option == "end-at":
                lines = lines[: index + 1]
            else:
                lines = lines[:index]
        assert any(line.strip() for line in lines), (page, path)
        if options.get("language") == "python":
            dedent = int(options.get("dedent", "0"))
            snippet = "".join(
                line[dedent:] if line.strip() else line for line in lines
            )
            ast.parse(snippet, filename=f"{page}:{path}")
