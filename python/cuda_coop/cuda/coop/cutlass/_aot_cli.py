# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command-line capture and selection for CUTLASS provider AOT packs."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

from . import aot


class _WorkloadExit(Exception):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


class _CommandError(ValueError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cuda-coop-aot",
        description="Capture, select, and inspect CUTLASS provider AOT packs.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    capture_parser = subparsers.add_parser(
        "capture",
        help="run a Python workload and atomically capture its provider bundles",
    )
    capture_parser.add_argument("--output", required=True, type=Path)
    capture_parser.add_argument("--name")
    capture_parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Python script or -m module command following --",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="run a Python workload with an explicit provider pack",
    )
    run_parser.add_argument("--pack", required=True, type=Path)
    run_parser.add_argument(
        "--mode",
        choices=("auto", "required", "off"),
        default="auto",
    )
    run_parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Python script or -m module command following --",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="validate a provider pack and emit structured JSON",
    )
    inspect_parser.add_argument("pack", type=Path)
    return parser


def _command_error(message: str) -> NoReturn:
    raise _CommandError(message)


def _strip_separator(command: Sequence[str]) -> list[str]:
    result = list(command)
    if result and result[0] == "--":
        result.pop(0)
    if not result:
        _command_error("a Python workload command is required after --")
    return result


def _same_interpreter(command: str) -> bool:
    resolved = shutil.which(command)
    if resolved is None:
        return False
    try:
        return os.path.samefile(resolved, sys.executable)
    except OSError:
        return os.path.realpath(resolved) == os.path.realpath(sys.executable)


def _python_workload(command: Sequence[str]) -> tuple[str, str, list[str]]:
    arguments = _strip_separator(command)
    interpreter = arguments.pop(0)
    if not _same_interpreter(interpreter):
        _command_error(
            "the workload must use the same Python interpreter as cuda-coop-aot"
        )
    if not arguments:
        _command_error("the Python workload is missing a script or -m module")
    if arguments[0] == "-m":
        if len(arguments) < 2 or not arguments[1] or arguments[1].startswith("-"):
            _command_error("-m requires a non-empty Python module name")
        return "module", arguments[1], arguments[2:]
    if arguments[0].startswith("-"):
        _command_error(
            "only 'python SCRIPT ...' and 'python -m MODULE ...' are supported"
        )
    return "script", arguments[0], arguments[1:]


def _system_exit_code(exception: SystemExit) -> int:
    code = exception.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def _run_python_workload(command: Sequence[str]) -> int:
    kind, target, arguments = _python_workload(command)
    original_argv = sys.argv
    original_path = sys.path
    original_path_contents = list(sys.path)
    try:
        if kind == "module":
            sys.argv = [target, *arguments]
            sys.path[0:1] = [os.getcwd()]
            try:
                runpy.run_module(target, run_name="__main__", alter_sys=True)
            except SystemExit as exception:
                return _system_exit_code(exception)
        else:
            script = Path(target).expanduser().resolve()
            if not script.is_file():
                _command_error(f"Python workload script does not exist: {target}")
            sys.argv = [str(script), *arguments]
            sys.path[0:1] = [str(script.parent)]
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as exception:
                return _system_exit_code(exception)
    finally:
        sys.argv = original_argv
        sys.path = original_path
        sys.path[:] = original_path_contents
    return 0


def _run_capture(arguments: argparse.Namespace) -> int:
    try:
        with aot.capture(arguments.output, name=arguments.name) as captured:
            code = _run_python_workload(arguments.command)
            if code:
                raise _WorkloadExit(code)
    except _WorkloadExit as exception:
        return exception.code
    result = captured.result
    print(
        f"Captured {len(result.entries)} provider bundle(s) to {result.path}.",
        file=sys.stderr,
    )
    return 0


def _run_with_pack(arguments: argparse.Namespace) -> int:
    with aot.use(arguments.pack, mode=arguments.mode):
        return _run_python_workload(arguments.command)


def _run_inspect(arguments: argparse.Namespace) -> int:
    info = aot.inspect(arguments.pack)
    payload = asdict(info)
    payload["path"] = os.fspath(info.path)
    payload["artifact_bytes"] = info.artifact_bytes
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
        if arguments.action == "capture":
            return _run_capture(arguments)
        if arguments.action == "run":
            return _run_with_pack(arguments)
        if arguments.action == "inspect":
            return _run_inspect(arguments)
    except (aot.PackError, _CommandError) as exception:
        parser.error(str(exception))
    raise AssertionError(f"unhandled cuda-coop-aot action: {arguments.action}")


if __name__ == "__main__":
    raise SystemExit(main())
