# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Capture and use CUTLASS provider packs while running a Python workload."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from cutlass.base_dsl.common import DSLRuntimeError

from . import _aot_pack, aot


def _run_workload(
    command: list[str],
    parser: argparse.ArgumentParser,
    *,
    from_command_line: bool,
) -> None:
    if command[:1] == ["--"]:
        command = command[1:]
    if len(command) < 2:
        parser.error("expected -- python SCRIPT or -- python -m MODULE")
    interpreter = shutil.which(command[0])
    if (
        not interpreter
        or not Path(interpreter).samefile(sys.executable)
        or Path(interpreter).parent.resolve()
        != Path(sys.executable).parent.resolve()
    ):
        parser.error("the workload must use the cuda-coop-aot interpreter")

    args = command[1:]
    is_module = args[0] == "-m"
    if is_module:
        args = args[1:]
    if not args or not args[0] or args[0].startswith("-"):
        parser.error("expected a Python module name or script path")

    target = args[0] if is_module else str(Path(args[0]).resolve())
    if not is_module and not Path(target).is_file():
        parser.error(f"Python workload script does not exist: {target}")
    previous_argv, previous_path = sys.argv, sys.path
    try:
        sys.argv = [target, *args[1:]]
        sys.path = previous_path.copy()
        if not getattr(sys.flags, "safe_path", sys.flags.isolated):
            if from_command_line:
                del sys.path[:1]  # Replace the launcher's import directory.
            entry_dir = Path.cwd() if is_module else Path(target).parent
            sys.path.insert(0, str(entry_dir))
        if is_module:
            runpy.run_module(target, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(target, run_name="__main__")
    except SystemExit as error:
        # A successful explicit exit still finishes and publishes the capture.
        if error.code is not None and not (
            isinstance(error.code, int) and error.code == 0
        ):
            raise
    finally:
        sys.argv, sys.path = previous_argv, previous_path


@contextmanager
def _use_pack(
    pack: Path, mode: Literal["auto", "required", "off"]
) -> Iterator[None]:
    # Newly spawned Python processes cannot inherit the selection context.
    path = pack.expanduser().resolve()
    environment = {
        _aot_pack.PACK_PATH_ENV: str(path),
        _aot_pack.PACK_MODE_ENV: mode,
    }
    previous = {key: os.environ.get(key) for key in environment}
    with aot.use(path, mode=mode):
        try:
            os.environ.update(environment)
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def main(argv: Sequence[str] | None = None) -> int:
    """Run a script or module under pack selection, or inspect a saved pack."""

    parser = argparse.ArgumentParser(prog="cuda-coop-aot", description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    capture = commands.add_parser("capture", help="capture providers in a pack")
    capture.add_argument("--output", required=True, type=Path)
    capture.add_argument("--name")
    run = commands.add_parser("run", help="run a workload with a saved pack")
    run.add_argument("--pack", required=True, type=Path)
    run.add_argument(
        "--mode", choices=("auto", "required", "off"), default="auto"
    )
    for subparser in (capture, run):
        subparser.add_argument(
            "command",
            nargs=argparse.REMAINDER,
            help="-- python SCRIPT [args] or -- python -m MODULE [args]",
        )
    inspect = commands.add_parser(
        "inspect", help="validate a pack and print JSON"
    )
    inspect.add_argument("pack", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.action == "inspect":
            info = aot.inspect(args.pack)
            print(
                json.dumps(
                    {**asdict(info), "artifact_bytes": info.artifact_bytes},
                    indent=2,
                )
            )
        elif args.action == "capture":
            with aot.capture(args.output, name=args.name):
                _run_workload(
                    args.command, parser, from_command_line=argv is None
                )
        else:
            with _use_pack(args.pack, args.mode):
                _run_workload(
                    args.command, parser, from_command_line=argv is None
                )
    except (aot.PackError, DSLRuntimeError) as error:
        # CuTe wraps errors from its trace-finalize hooks.
        cause: BaseException | None = error
        seen: set[int] = set()
        while cause is not None and id(cause) not in seen:
            if isinstance(cause, aot.PackError):
                print(f"{parser.prog}: {cause}", file=sys.stderr)
                return 1
            seen.add(id(cause))
            cause = cause.__cause__
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
