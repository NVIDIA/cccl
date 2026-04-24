# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

import difflib
from argparse import ArgumentParser
from pathlib import Path
from re import Match, Pattern
from re import compile as re_compile
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Sequence


class ReplacementError(Exception):
    def __init__(self, path: Path, msg: str) -> None:
        self.path = path
        self.msg = msg


class Replacement:
    __slots__ = "ignore_re", "on_file_change", "pattern", "repl"

    def __init__(
        self,
        pattern: str,
        repl: str | Callable[[Path, Match], str],
        pragma_keyword: str,
        flags: int = 0,
        on_file_change: Callable[[Path, str], str] | None = None,
    ) -> None:
        self.pattern = re_compile(pattern, flags=flags)
        self.ignore_re = self._make_ignore_re(pragma_keyword)
        self.repl = self._make_repl(repl)
        if on_file_change is None:
            on_file_change = self._default_on_file_change

        self.on_file_change = on_file_change

    @staticmethod
    def _default_on_file_change(_p: Path, s: str) -> str:
        return s

    @staticmethod
    def _make_ignore_re(kwd: str) -> Pattern:
        base_ignore_pat = rf"cccl\-lint:.*no\-{kwd}"
        return re_compile(rf"(//\s*{base_ignore_pat}|/\*\s*{base_ignore_pat}\s*\*/)")

    @staticmethod
    def _sanitize_repl(
        repl: str | Callable[[Path, Match], str],
    ) -> Callable[[Path, Match], str]:
        if callable(repl):
            return repl

        def str_repl_wrap(_path: Path, re_match: Match) -> str:
            return re_match.expand(repl)

        return str_repl_wrap

    def _make_repl(
        self, repl: str | Callable[[Match], str]
    ) -> Callable[[Path], Callable[[Match], str]]:
        repl = self._sanitize_repl(repl)

        def closure(path: Path) -> Callable[[Match], str]:
            def repl_wrap(re_match: Match) -> str:
                if self.is_silenced(re_match):
                    return re_match[0]
                return repl(path, re_match)

            return repl_wrap

        return closure

    def replace(self, path: Path, text: str) -> str:
        return self.pattern.sub(self.repl(path), text)

    def is_silenced(self, re_match: Match) -> bool:
        string = re_match.string
        rest_begin = re_match.end()
        # Only search for the ignore regex until the end of the line
        silence_found = self.ignore_re.search(
            string, rest_begin, string.find("\n", rest_begin)
        )
        return silence_found is not None


class RegexReplacement:
    __slots__ = (
        "allowed_suffixes",
        "description",
        "dry_run",
        "files",
        "replacements",
        "verbose",
    )

    class AllSuffixesImpl:
        def __contains__(self, item: Any) -> bool:
            return True

    AllSuffixes: Final = AllSuffixesImpl()

    CppSuffixes: Final = {
        ".h",
        ".hpp",
        ".inl",
        ".cc",
        ".cpp",
        ".cxx",
        ".cu",
        ".cuh",
        ".cuinl",
    }

    def __init__(
        self,
        description: str,
        replacements: Sequence[Replacement],
        allowed_suffixes: Container[str] | None = None,
    ) -> None:
        assert len(replacements)
        self.description = description
        self.replacements = replacements
        self.files = []
        self.dry_run = False
        self.verbose = False
        if allowed_suffixes is None:
            allowed_suffixes = self.CppSuffixes

        self.allowed_suffixes = allowed_suffixes

    def parse_args(self) -> None:
        parser = ArgumentParser(description=self.description)
        parser.add_argument(
            "files",
            nargs="+",
            type=Path,
            help=(
                "Path(s) to the file(s) or directories to be searched. "
                "Directories are searched recursively"
            ),
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Enable verbose output"
        )
        parser.add_argument(
            "--dry-run", action="store_true", help="Whether to do a dry run"
        )
        opts = parser.parse_args()

        # convert a directory to recursive list of files
        if len(opts.files) == 1 and opts.files[0].is_dir():
            opts.files = list(opts.files[0].rglob("*"))

        self.files = opts.files
        self.dry_run = opts.dry_run
        self.verbose = opts.verbose

    def handle_file(self, file_path: Path) -> tuple[bool, list[ReplacementError]]:
        try:
            orig_text = file_path.read_text()
        except UnicodeDecodeError as e:
            m = f"Unable to decode {file_path}"
            raise ValueError(m) from e

        old_text = orig_text
        new_text = old_text
        changed = False
        errors = []
        for repl in self.replacements:
            try:
                new_text = repl.replace(file_path, old_text)
            except ReplacementError as exn:
                errors.append(exn)

            if new_text == old_text:
                continue

            try:
                new_text = repl.on_file_change(file_path, new_text)
            except ReplacementError as exn:
                errors.append(exn)
            else:
                changed = True
                old_text = new_text

        if changed:
            if self.verbose:
                diff = difflib.unified_diff(
                    orig_text.splitlines(),
                    new_text.splitlines(),
                    fromfile="before",
                    tofile="after",
                )
                diff = list(diff)[3:]
                print("\n".join(diff))  # noqa: T201
            if not self.dry_run:
                file_path.write_text(new_text)
        return changed, errors

    def vprint(self, *args: Any, **kwargs: Any) -> None:
        if self.verbose:
            print(*args, **kwargs)  # noqa: T201

    def main(self) -> int:
        self.parse_args()

        modified = []
        allowed_suff = self.allowed_suffixes
        ret = 0
        for file_path in self.files:
            if (file_path.suffix not in allowed_suff) or (not file_path.is_file()):
                self.vprint("skipping:     ", file_path)
                continue

            self.vprint("processing:   ", file_path)
            changed, errata = self.handle_file(file_path)
            if changed:
                self.vprint("found changes:", file_path)
                modified.append(file_path)
            for err in errata:
                print(  # noqa: T201
                    "--", err.path, "requires manual intervention:", err.msg
                )
                ret = 1

        mod_str = "would have modified:" if self.dry_run else "modified:"
        for file_path in modified:
            print(mod_str, file_path)  # noqa: T201
        return ret
