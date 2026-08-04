# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run a libcudacxx pretty-printer scenario under LLDB or GDB."""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

_MARKER_EDGE = "=" * 15
_MARKER_PATTERN = re.compile(
    rf"^{re.escape(_MARKER_EDGE)} (?P<section>.+) (?P<kind>begin|end) {re.escape(_MARKER_EDGE)}$"
)
_LLDB_ECHO_PATTERN = re.compile(r"^\s*\(lldb\)\s")
_GDB_VALUE_PREFIX_PATTERN = re.compile(r"^\s*\$\d+ = ")
_NONZERO_HEX_PATTERN = re.compile(r"\b0x(?!0+\b)[0-9a-fA-F]+\b")


class HarnessError(RuntimeError):
    """Report invalid test input or debugger output."""


class DebuggerError(RuntimeError):
    """Report a debugger launch, timeout, or exit failure."""


class Debugger(StrEnum):
    LLDB = "lldb"
    GDB = "gdb"


@dataclass(frozen=True)
class Case:
    breakpoint: str
    frame: int
    section: str
    expression: str


class CaseAction(argparse.Action):
    """Parse and validate one four-part ``--case`` option."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Sequence[str],
        option_string: str | None = None,
    ) -> None:
        """Append one parsed case to the argument namespace.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Parser handling the command line.
        namespace : argparse.Namespace
            Namespace receiving parsed cases.
        values : Sequence[str]
            Breakpoint, frame, section, and expression values.
        option_string : str or None
            Option spelling that supplied the values.

        Raises
        ------
        SystemExit
            If the frame, breakpoint, section, or expression is invalid, or if
            the section name is duplicated.
        """
        breakpoint, raw_frame, section, expression = values
        try:
            frame = int(raw_frame)
        except ValueError:
            parser.error(f"invalid caller frame index {raw_frame!r}")
        if frame < 0:
            parser.error(f"caller frame index must be nonnegative: {frame}")
        for label, value in (
            ("breakpoint", breakpoint),
            ("section", section),
            ("expression", expression),
        ):
            if not value or "\n" in value or "\r" in value:
                parser.error(f"{label} must be a nonempty single line")

        cases: list[Case] = getattr(namespace, self.dest) or []
        if any(case.section == section for case in cases):
            parser.error(f"duplicate section name: {section}")
        cases.append(Case(breakpoint, frame, section, expression))
        setattr(namespace, self.dest, cases)


def marker(section: str, kind: str) -> str:
    """Build an exact marker for a captured section.

    Parameters
    ----------
    section : str
        Unique name of the output section.
    kind : str
        Marker kind, either ``begin`` or ``end``.

    Returns
    -------
    str
        Complete marker line expected in debugger output.
    """
    return f"{_MARKER_EDGE} {section} {kind} {_MARKER_EDGE}"


class DebuggerAdapter(ABC):
    """Provide debugger-specific command and transcript hooks.

    Parameters
    ----------
    executable : Path
        Debugger executable path.
    formatter_init : Path
        Pretty-printer entry-point path.
    program : Path
        Scenario executable path.
    """

    kind: Debugger

    def __init__(self, executable: Path, formatter_init: Path, program: Path) -> None:
        """Store paths shared by debugger-specific operations.

        Parameters
        ----------
        executable : Path
            Debugger executable path.
        formatter_init : Path
            Pretty-printer entry-point path.
        program : Path
            Scenario executable path.
        """
        self.executable = executable
        self.formatter_init = formatter_init
        self.program = program

    def generate_commands(self, cases: Sequence[Case]) -> str:
        """Generate commands for an ordered list of cases.

        Parameters
        ----------
        cases : Sequence[Case]
            Cases to execute in order.

        Returns
        -------
        str
            Complete debugger command-file contents.

        Raises
        ------
        HarnessError
            If a completed breakpoint group is reopened later.
        """
        closed_stops: set[tuple[str, int]] = set()
        previous_stop: tuple[str, int] | None = None
        for case in cases:
            stop = (case.breakpoint, case.frame)
            if stop == previous_stop:
                continue
            if stop in closed_stops:
                raise HarnessError(
                    f"breakpoint group {case.breakpoint!r} at frame {case.frame} was reopened"
                )
            if previous_stop is not None:
                closed_stops.add(previous_stop)
            previous_stop = stop
        return self._generate_commands(cases)

    @abstractmethod
    def _generate_commands(self, cases: Sequence[Case]) -> str:
        """Generate commands for validated cases.

        Parameters
        ----------
        cases : Sequence[Case]
            Cases to execute in order.

        Returns
        -------
        str
            Complete debugger command-file contents.
        """
        raise NotImplementedError

    @abstractmethod
    def command(self, command_file: Path) -> list[str]:
        """Build the debugger subprocess argument list.

        Parameters
        ----------
        command_file : Path
            Generated debugger command-file path.

        Returns
        -------
        list[str]
            Subprocess arguments for this debugger.
        """
        raise NotImplementedError

    def include_transcript_line(self, line: str) -> bool:
        """Return whether a line inside a marked section should be retained.

        Parameters
        ----------
        line : str
            Transcript line inside a marked section.

        Returns
        -------
        bool
            ``True`` when the line belongs in normalized output.
        """
        return True

    def normalize_line(self, line: str) -> str:
        """Apply debugger-specific normalization to one output line.

        Parameters
        ----------
        line : str
            Extracted debugger output line.

        Returns
        -------
        str
            Line after debugger-specific normalization.
        """
        return line


class GDB(DebuggerAdapter):
    """Provide GDB-specific pretty-printer test behavior."""

    kind = Debugger.GDB

    def _generate_commands(self, cases: Sequence[Case]) -> str:
        """Generate a GDB command file for ordered cases.

        Parameters
        ----------
        cases : Sequence[Case]
            Cases to execute in order.

        Returns
        -------
        str
            Complete GDB command-file contents.
        """
        lines = [
            "set pagination off",
            "set print pretty on",
            "set print array-indexes on",
            "set debuginfod enabled off",
            f"source {self.formatter_init}",
        ]
        seen_breakpoints: set[str] = set()
        for case in cases:
            if case.breakpoint in seen_breakpoints:
                continue
            lines.append(f"break {case.breakpoint}")
            seen_breakpoints.add(case.breakpoint)
        lines.append("run")
        previous_stop: tuple[str, int] | None = None
        for case in cases:
            stop = (case.breakpoint, case.frame)
            if previous_stop is not None and stop != previous_stop:
                lines.append("continue")
            if stop != previous_stop:
                lines.append(f"frame {case.frame}")
            previous_stop = stop
            begin = marker(case.section, "begin")
            end = marker(case.section, "end")
            expression = repr(case.expression)
            lines.extend(
                [
                    f"python print({begin!r})",
                    "python",
                    "try:",
                    f"    print(gdb.execute('print ' + {expression}, from_tty=True, to_string=True), end='')",
                    "except Exception as error:",
                    "    print(error)",
                    "end",
                    f"python print({end!r})",
                ]
            )
        return "\n".join(lines) + "\n"

    def command(self, command_file: Path) -> list[str]:
        """Build the GDB subprocess argument list.

        Parameters
        ----------
        command_file : Path
            Generated GDB command-file path.

        Returns
        -------
        list[str]
            GDB subprocess arguments.
        """
        return [
            str(self.executable),
            "--quiet",
            "--batch",
            "--nx",
            "--command",
            str(command_file),
            str(self.program),
        ]

    def normalize_line(self, line: str) -> str:
        """Remove GDB value-history prefixes from an output line.

        Parameters
        ----------
        line : str
            Extracted GDB output line.

        Returns
        -------
        str
            Line without a leading ``$N =`` prefix.
        """
        return _GDB_VALUE_PREFIX_PATTERN.sub("", line)


class LLDB(DebuggerAdapter):
    """Provide LLDB-specific pretty-printer test behavior."""

    kind = Debugger.LLDB

    def _generate_commands(self, cases: Sequence[Case]) -> str:
        """Generate an LLDB command file for ordered cases.

        Parameters
        ----------
        cases : Sequence[Case]
            Cases to execute in order.

        Returns
        -------
        str
            Complete LLDB command-file contents.
        """
        lines = [f'command script import "{self.formatter_init}"']
        seen_breakpoints: set[str] = set()
        for case in cases:
            if case.breakpoint in seen_breakpoints:
                continue
            lines.append(f"breakpoint set --name {case.breakpoint}")
            seen_breakpoints.add(case.breakpoint)
        lines.append("run")
        previous_stop: tuple[str, int] | None = None
        for case in cases:
            stop = (case.breakpoint, case.frame)
            if previous_stop is not None and stop != previous_stop:
                lines.append("continue")
            if stop != previous_stop:
                lines.append(f"frame select {case.frame}")
            previous_stop = stop
            begin = marker(case.section, "begin")
            end = marker(case.section, "end")
            debugger_command = f"dwim-print -- {case.expression}"
            lines.extend(
                [
                    f"script print({begin!r})",
                    "script result = lldb.SBCommandReturnObject(); "
                    f"status = lldb.debugger.GetCommandInterpreter().HandleCommand({debugger_command!r}, result); "
                    "print(result.GetOutput(), end=''); print(result.GetError(), end='')",
                    f"script print({end!r})",
                ]
            )
        return "\n".join(lines) + "\n"

    def command(self, command_file: Path) -> list[str]:
        """Build the LLDB subprocess argument list.

        Parameters
        ----------
        command_file : Path
            Generated LLDB command-file path.

        Returns
        -------
        list[str]
            LLDB subprocess arguments.
        """
        return [
            str(self.executable),
            "--batch",
            "--no-lldbinit",
            "--source",
            str(command_file),
            str(self.program),
        ]

    def include_transcript_line(self, line: str) -> bool:
        """Exclude LLDB prompt and command-echo lines from marked output.

        Parameters
        ----------
        line : str
            Transcript line inside a marked section.

        Returns
        -------
        bool
            ``False`` for LLDB prompt or command-echo lines.
        """
        return _LLDB_ECHO_PATTERN.match(line) is None


def extract_sections(
    transcript: str, section_order: Sequence[str], debugger: DebuggerAdapter
) -> str:
    """Extract and validate marked sections from a debugger transcript.

    Parameters
    ----------
    transcript : str
        Complete combined debugger output.
    section_order : Sequence[str]
        Expected section names in manifest order.
    debugger : DebuggerAdapter
        Adapter for the debugger that produced the transcript.

    Returns
    -------
    str
        Marked sections concatenated in manifest order.

    Raises
    ------
    HarnessError
        If markers are unexpected, missing, duplicated, nested, mismatched, or
        unterminated.
    """
    expected_sections = set(section_order)
    captured: dict[str, list[str]] = {}
    active_section: str | None = None

    for line in transcript.splitlines():
        match = _MARKER_PATTERN.fullmatch(line)
        if not match:
            if active_section is None:
                continue
            if not debugger.include_transcript_line(line):
                continue
            captured[active_section].append(line)
            continue

        section = match.group("section")
        if section not in expected_sections:
            raise HarnessError(f"unexpected marked section: {section}")

        kind = match.group("kind")

        match kind:
            case "begin":
                if active_section is not None:
                    raise HarnessError(
                        f"nested section {section!r} inside {active_section!r}"
                    )
                if section in captured:
                    raise HarnessError(f"duplicate marked section: {section}")
                captured[section] = [line]
                active_section = section
            case "end":
                if active_section is None:
                    raise HarnessError(f"end marker without begin marker: {section}")
                if active_section != section:
                    raise HarnessError(
                        f"mismatched end marker for {section!r}; expected {active_section!r}"
                    )
                captured[section].append(line)
                active_section = None
            case _:
                raise HarnessError(f"invalid marker kind: {kind}")

    if active_section is not None:
        raise HarnessError(f"unterminated marked section: {active_section}")
    missing = [section for section in section_order if section not in captured]
    if missing:
        raise HarnessError(f"missing marked sections: {', '.join(missing)}")

    lines: list[str] = []
    for section in section_order:
        lines.extend(captured[section])
    return "\n".join(lines) + "\n"


def normalize_output(output: str, debugger: DebuggerAdapter) -> str:
    """Normalize unstable values while preserving output structure.

    Parameters
    ----------
    output : str
        Extracted marked output.
    debugger : DebuggerAdapter
        Adapter that applies debugger-specific line normalization.

    Returns
    -------
    str
        Output with unstable addresses and debugger prefixes normalized.
    """
    normalized_lines: list[str] = []
    for line in output.splitlines():
        line = debugger.normalize_line(line.rstrip())
        # Some debuggers may print C++98 style > > for multiple templates.
        line = re.sub(r">\s+>", ">>", line)
        line = _NONZERO_HEX_PATTERN.sub("<address>", line)
        normalized_lines.append(line)
    return "\n".join(normalized_lines) + "\n"


def compare_expected(
    actual: str, expected: str, debugger: DebuggerAdapter, scenario: str
) -> None:
    """Compare normalized output with its checked-in golden text.

    Parameters
    ----------
    actual : str
        Normalized debugger output.
    expected : str
        Checked-in golden output.
    debugger : DebuggerAdapter
        Adapter for the debugger that produced the output.
    scenario : str
        Scenario name used in diagnostics.

    Raises
    ------
    HarnessError
        If actual and expected output differ.
    """
    if actual == expected:
        return
    difference = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=f"{scenario}/{debugger.kind}.expected",
            tofile=f"{scenario}/{debugger.kind}.actual",
        )
    )
    raise HarnessError(
        f"{debugger.kind} pretty-printer output mismatch for {scenario}:\n{difference}"
    )


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    """Parse debugger configuration and ordered case definitions.

    Parameters
    ----------
    arguments : Sequence[str] or None
        Command-line arguments, or ``None`` to use ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--debugger", type=Debugger, choices=Debugger, required=True)
    parser.add_argument("--debugger-executable", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--formatter-init", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--update-expected", action="store_true")
    parser.add_argument(
        "--case",
        dest="cases",
        nargs=4,
        action=CaseAction,
        default=None,
        required=True,
    )
    return parser.parse_args(arguments)


def _create_debugger(args: argparse.Namespace) -> DebuggerAdapter:
    """Create the debugger adapter selected by command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed runner arguments.

    Returns
    -------
    DebuggerAdapter
        Configured adapter for the selected debugger.

    Raises
    ------
    HarnessError
        If the selected debugger is unsupported.
    """
    match args.debugger:
        case Debugger.LLDB:
            return LLDB(args.debugger_executable, args.formatter_init, args.program)
        case Debugger.GDB:
            return GDB(args.debugger_executable, args.formatter_init, args.program)
        case _:
            raise HarnessError(f"unsupported debugger: {args.debugger}")


def _run_debugger(
    args: argparse.Namespace, debugger: DebuggerAdapter, command_file: Path
) -> str:
    """Run the debugger and persist its complete transcript.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed runner arguments.
    debugger : DebuggerAdapter
        Configured debugger adapter.
    command_file : Path
        Generated debugger command-file path.

    Returns
    -------
    str
        Complete combined debugger output.

    Raises
    ------
    DebuggerError
        If the debugger cannot launch, times out, or exits with a nonzero status.
    OSError
        If the transcript cannot be written.
    """
    try:
        completed = subprocess.run(
            debugger.command(command_file),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as error:
        if isinstance(error.stdout, str):
            args.output_log.write_text(error.stdout)
        raise DebuggerError(
            f"{debugger.kind} timed out after {args.timeout:g} seconds"
        ) from error
    except OSError as error:
        raise DebuggerError(f"failed to launch {debugger.kind}: {error}") from error

    args.output_log.write_text(completed.stdout)
    if completed.returncode != 0:
        raise DebuggerError(
            f"{debugger.kind} exited with status {completed.returncode}"
        )
    return completed.stdout


def _match_output(
    args: argparse.Namespace,
    debugger: DebuggerAdapter,
    cases: Sequence[Case],
    transcript: str,
) -> None:
    """Extract, normalize, and compare or update debugger output.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed runner arguments.
    debugger : DebuggerAdapter
        Configured debugger adapter.
    cases : Sequence[Case]
        Validated cases in manifest order.
    transcript : str
        Complete combined debugger output.

    Raises
    ------
    HarnessError
        If marked output is invalid or differs from the golden.
    OSError
        If the golden file cannot be read or updated.
    """
    extracted = extract_sections(transcript, [case.section for case in cases], debugger)
    actual = normalize_output(extracted, debugger)
    if args.update_expected:
        args.expected.write_text(actual)
        return
    compare_expected(
        actual,
        args.expected.read_text(),
        debugger,
        args.expected.parent.name,
    )


def _report_error(
    args: argparse.Namespace,
    debugger: DebuggerAdapter,
    command_file: Path,
    error: Exception,
) -> None:
    """Report a debugger or output-matching failure with artifact paths.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed runner arguments.
    debugger : DebuggerAdapter
        Configured debugger adapter.
    command_file : Path
        Generated debugger command-file path.
    error : Exception
        Failure being reported.
    """
    scenario = args.expected.parent.name
    print(
        f"error: {debugger.kind} pretty-printer test for {scenario}: {error}",
        file=sys.stderr,
    )
    print(f"debugger commands: {command_file}", file=sys.stderr)
    if args.output_log.exists():
        print(f"complete transcript: {args.output_log}", file=sys.stderr)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one debugger pretty-printer test from command-line arguments.

    Parameters
    ----------
    arguments : Sequence[str] or None
        Command-line arguments, or ``None`` to use ``sys.argv``.

    Returns
    -------
    int
        Zero on success and one for handled debugger or matching failures.

    Raises
    ------
    HarnessError
        If case setup is invalid.
    OSError
        If setup artifacts or golden files cannot be accessed.
    """
    args = _parse_arguments(arguments)
    debugger = _create_debugger(args)
    commands = debugger.generate_commands(args.cases)
    command_file = args.output_log.with_suffix(".commands")
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.unlink(missing_ok=True)
    command_file.write_text(commands)

    try:
        transcript = _run_debugger(args, debugger, command_file)
    except DebuggerError as error:
        _report_error(args, debugger, command_file, error)
        return 1

    try:
        _match_output(args, debugger, args.cases, transcript)
    except HarnessError as error:
        _report_error(args, debugger, command_file, error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
