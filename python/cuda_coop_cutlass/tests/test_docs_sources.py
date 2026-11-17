# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check documentation sources for mistakes that Sphinx hides.

``docs/conf.py`` suppresses docutils warnings. A ``literalinclude`` with a
missing file or marker then disappears from the built page while the build
passes. An ``automodule`` directive in ``api.rst`` also changes the module of
every entry below it until ``currentmodule`` restores the public one.

These tests read the RST pages and the package docstrings as text. They need
a CCCL checkout. They never import the package or touch a GPU.
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import takewhile
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parents[1]
_DOCS_ROOT = _REPO_ROOT / "docs" / "python" / "coop_cutlass"
_SOURCE_ROOT = _PACKAGE_ROOT / "cuda" / "coop" / "cutlass"

if not _DOCS_ROOT.is_dir():
    pytest.skip(
        f"documentation checks need a CCCL checkout: {_DOCS_ROOT} is missing",
        allow_module_level=True,
    )

_LITERALINCLUDE = re.compile(r"^(\s*)\.\. literalinclude::(.*)$")
_OPTION = re.compile(r":([\w-]+):\s*(.*)")
_MARKER_OPTIONS = ("start-after", "start-at", "end-before", "end-at")
_UNIQUE_MARKER_OPTIONS = ("start-after", "end-before")
_GITHUB_URL = re.compile(r"https://github\.com/[^\s>]+")
_AUTODOC = re.compile(r"^\s*\.\. auto(module|class|function|method)::")
_PUBLIC_MODULE = ".. currentmodule:: cuda.coop.cutlass"
_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class _Include:
    """One ``literalinclude`` directive and the text written above it."""

    source: Path
    line: int
    target: str
    options: dict[str, str]
    text_above: str = field(repr=False)


def _parse_includes(
    text: str, source: Path, first_line: int = 1
) -> list[_Include]:
    """Return the ``literalinclude`` directives in RST ``text``.

    The file name sits on the directive line or, as in the package
    docstrings, on the line below it. Options follow until a blank line.
    """
    lines = text.splitlines()
    includes = []
    for index, line in enumerate(lines):
        directive = _LITERALINCLUDE.match(line)
        if directive is None:
            continue
        block = [directive[2].strip()]
        for follower in lines[index + 1 :]:
            entry = follower.strip()
            indent = len(follower) - len(follower.lstrip())
            if not entry or indent <= len(directive[1]):
                break
            block.append(entry)
        arguments = takewhile(lambda item: not _OPTION.match(item), block)
        includes.append(
            _Include(
                source=source,
                line=first_line + index,
                target=" ".join(filter(None, arguments)),
                options={
                    option[1]: option[2]
                    for option in map(_OPTION.match, block)
                    if option
                },
                text_above="\n".join(lines[:index]),
            )
        )
    return includes


def _docstrings(module: Path) -> Iterator[tuple[int, str]]:
    """Yield the first line number and raw text of each docstring."""
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    for node in ast.walk(tree):
        if isinstance(node, _DOCUMENTED):
            text = ast.get_docstring(node, clean=False)
            if text:
                yield node.body[0].lineno, text


def _page_includes(docs_root: Path) -> list[_Include]:
    return [
        include
        for page in sorted(docs_root.rglob("*.rst"))
        for include in _parse_includes(page.read_text(encoding="utf-8"), page)
    ]


def _docstring_includes(source_root: Path) -> list[_Include]:
    return [
        include
        for module in sorted(source_root.rglob("*.py"))
        for first_line, text in _docstrings(module)
        for include in _parse_includes(text, module, first_line)
    ]


def _marker_problems(lines: list[str], options: dict[str, str]) -> list[str]:
    """Describe the markers that Sphinx would fail to find in ``lines``.

    Sphinx takes the first line that contains the start text and looks for
    the end text in the lines that remain. It skips an ``end-before`` match
    on the first of those lines. A ``start-after`` or ``end-before`` marker
    must also be unique so that the include keeps pointing at one region.
    """
    problems = []
    start = 0  # Sphinx searches for the end text below this line number.
    for name in _MARKER_OPTIONS:
        if name not in options:
            continue
        marker = options[name]
        option = f":{name}: {marker!r}"
        hits = [n for n, line in enumerate(lines, 1) if marker in line]
        where = ", ".join(map(str, hits))
        if not hits:
            problems.append(f"{option} matches 0 lines")
        elif name in _UNIQUE_MARKER_OPTIONS and len(hits) > 1:
            problems.append(
                f"{option} matches {len(hits)} lines ({where}), expected 1"
            )
        elif name == "start-after":
            start = hits[0]
        elif name == "start-at":
            start = hits[0] - 1
        elif hits[-1] <= start + (name == "end-before"):
            problems.append(
                f"{option} matches 0 lines below the start marker "
                f"(found on line {where})"
            )
    return problems


def _include_problems(include: _Include, directory: Path) -> list[str]:
    """Check the file and markers of one directive read from ``directory``."""
    path = directory / include.target
    if not path.is_file():
        return [f"file not found: {os.path.normpath(path)}"]
    lines = path.read_text(encoding="utf-8").splitlines()
    return _marker_problems(lines, include.options)


def _link_problems(include: _Include) -> list[str]:
    """Require a GitHub link to the included file above a docstring include."""
    path = os.path.normpath(_DOCS_ROOT / include.target)
    repo_path = Path(os.path.relpath(path, _REPO_ROOT)).as_posix()
    links = _GITHUB_URL.findall(include.text_above)
    if any(link.endswith("/" + repo_path) for link in links):
        return []
    return [f"no GitHub link to {repo_path} above the directive"]


def _automodules_without_reset(text: str) -> list[str]:
    """List the ``automodule`` directives that leave their module current.

    ``.. currentmodule:: cuda.coop.cutlass`` must follow each one before the
    next autodoc directive. Entries below it otherwise document the private
    module path.
    """
    missing = []
    pending = None
    for number, line in enumerate(text.splitlines(), 1):
        if line.strip() == _PUBLIC_MODULE:
            pending = None
        elif directive := _AUTODOC.match(line):
            if pending:
                missing.append(pending)
            pending = None
            if directive[1] == "module":
                pending = f"line {number}: {line.strip()}"
    if pending:
        missing.append(pending)
    return missing


def _label(include: _Include) -> str:
    source = include.source.relative_to(_REPO_ROOT).as_posix()
    return f"{source}:{include.line}"


def _report(include: _Include, problems: list[str]) -> str:
    header = f"{_label(include)}: literalinclude {include.target}"
    return "\n  ".join([header, *problems])


_PAGE_INCLUDES = _page_includes(_DOCS_ROOT)
_DOCSTRING_INCLUDES = _docstring_includes(_SOURCE_ROOT)


def test_literalincludes_were_found() -> None:
    assert _PAGE_INCLUDES, f"no literalinclude under {_DOCS_ROOT}"
    assert _DOCSTRING_INCLUDES, f"no literalinclude under {_SOURCE_ROOT}"


@pytest.mark.parametrize("include", _PAGE_INCLUDES, ids=_label)
def test_page_literalinclude_finds_file_and_markers(include: _Include) -> None:
    problems = _include_problems(include, include.source.parent)
    assert not problems, _report(include, problems)


@pytest.mark.parametrize("include", _DOCSTRING_INCLUDES, ids=_label)
def test_docstring_literalinclude_finds_file_markers_and_link(
    include: _Include,
) -> None:
    # Docstrings render on api.rst, so Sphinx resolves their includes there.
    problems = _include_problems(include, _DOCS_ROOT) + _link_problems(include)
    assert not problems, _report(include, problems)


def test_api_automodules_restore_public_module() -> None:
    api = _DOCS_ROOT / "api.rst"
    missing = _automodules_without_reset(api.read_text(encoding="utf-8"))
    page = api.relative_to(_REPO_ROOT).as_posix()
    header = f"{page}: add '{_PUBLIC_MODULE}' after"
    assert not missing, "\n  ".join([header, *missing])


# The remaining tests run the checkers on small inputs with known mistakes.
_EXAMPLE_FILE = """\
# begin
value = 1
# end
value = 2
# end
# last
"""
_EXAMPLE_PAGE = """\
.. literalinclude:: example.py
   :start-at: value = 1
   :end-at: # end

.. literalinclude:: example.py
   :start-after: # begin
   :end-before: # end
.. literalinclude::
   example.py
   :language: python
   :start-after: # last
   :end-before: # begin

.. literalinclude:: example.py
   :start-at: value
   :end-before: # absent

.. literalinclude:: absent.py
"""


def test_include_checker_reports_broken_directives(tmp_path: Path) -> None:
    (tmp_path / "example.py").write_text(_EXAMPLE_FILE, encoding="utf-8")
    includes = _parse_includes(_EXAMPLE_PAGE, tmp_path / "page.rst")
    problems = [_include_problems(include, tmp_path) for include in includes]

    assert [include.line for include in includes] == [1, 5, 8, 14, 18]
    assert problems == [
        [],
        [":end-before: '# end' matches 2 lines (3, 5), expected 1"],
        [
            ":end-before: '# begin' matches 0 lines below the start marker "
            "(found on line 1)"
        ],
        [":end-before: '# absent' matches 0 lines"],
        [f"file not found: {tmp_path / 'absent.py'}"],
    ]


_EXAMPLE_MODULE = '''\
def scan():
    """Scan values.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/scan.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/examples/scan.py
        :language: python
    """
'''


def test_docstring_checker_requires_link_to_included_file(
    tmp_path: Path,
) -> None:
    module = tmp_path / "example.py"
    module.write_text(_EXAMPLE_MODULE, encoding="utf-8")
    (linked,) = _docstring_includes(tmp_path)
    module.write_text(
        _EXAMPLE_MODULE.replace("scan.py>", "reduce.py>"), encoding="utf-8"
    )
    (unlinked,) = _docstring_includes(tmp_path)

    assert (linked.line, linked.options) == (7, {"language": "python"})
    assert _link_problems(linked) == []
    assert _link_problems(unlinked) == [
        "no GitHub link to python/cuda_coop_cutlass/examples/scan.py "
        "above the directive"
    ]


_API_SECTION = f"""\
.. automodule:: cuda.coop.cutlass._temp_storage
   :no-members:

{_PUBLIC_MODULE}

.. autoclass:: TempStorage
"""
_API_AUTOMODULE = "line 1: .. automodule:: cuda.coop.cutlass._temp_storage"


@pytest.mark.parametrize(
    "text,missing",
    [
        pytest.param(_API_SECTION, [], id="reset-follows"),
        pytest.param(
            _API_SECTION.replace(_PUBLIC_MODULE, ""),
            [_API_AUTOMODULE],
            id="reset-missing",
        ),
        pytest.param(
            _API_SECTION.replace(_PUBLIC_MODULE, _PUBLIC_MODULE + "._types"),
            [_API_AUTOMODULE],
            id="reset-names-another-module",
        ),
        pytest.param(
            _API_SECTION.partition(_PUBLIC_MODULE)[0],
            [_API_AUTOMODULE],
            id="reset-missing-at-end-of-page",
        ),
        pytest.param(
            ".. automodule:: cuda.coop.cutlass._group_reduce\n" + _API_SECTION,
            ["line 1: .. automodule:: cuda.coop.cutlass._group_reduce"],
            id="stacked-automodules-share-one-reset",
        ),
    ],
)
def test_automodule_checker_requires_reset(
    text: str, missing: list[str]
) -> None:
    assert _automodules_without_reset(text) == missing
