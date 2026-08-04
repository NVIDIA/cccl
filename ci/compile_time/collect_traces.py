#!/usr/bin/env python3

import argparse
import shutil
import tempfile
from pathlib import Path

TRACE_SUFFIXES = (".o.json", ".obj.json")


def parse_input(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("inputs must use LABEL=PATH")
    label_path = Path(label)
    if label in {".", ".."} or label_path.is_absolute() or label_path.parts != (label,):
        raise argparse.ArgumentTypeError("input labels must be single directory names")
    return label, Path(path)


def trace_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file() and path.name.endswith(TRACE_SUFFIXES)
    )


def replace_output(output: Path, collected: list[tuple[Path, Path]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        for source, destination in collected:
            staged_destination = staging / destination.relative_to(output)
            staged_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged_destination)

        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect object-adjacent NVCC device-time traces."
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        required=True,
        type=parse_input,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    labels = [label for label, _ in args.inputs]
    if len(labels) != len(set(labels)):
        parser.error("input labels must be unique")

    if args.output.is_symlink():
        parser.error("refusing to use a symbolic link as --output")
    output = args.output.resolve(strict=False)
    if output == Path(output.anchor):
        parser.error("refusing to use a filesystem root as --output")
    if output.exists() and not output.is_dir():
        parser.error(f"output is not a directory: {args.output}")

    collected: list[tuple[Path, Path]] = []
    for label, input_path in args.inputs:
        try:
            root = input_path.resolve(strict=True)
        except OSError as error:
            parser.error(f"cannot read input {input_path}: {error}")
        if not root.is_dir():
            parser.error(f"input is not a directory: {input_path}")
        if output == root or output in root.parents or root in output.parents:
            parser.error(
                f"input and output paths must not overlap: {input_path} and {args.output}"
            )
        collected.extend(
            (trace, output / label / trace.relative_to(root))
            for trace in trace_paths(root)
        )

    if not collected:
        parser.error("no object-adjacent device-time traces found")

    replace_output(output, collected)

    print(f"collected {len(collected)} trace(s) under {output}")


if __name__ == "__main__":
    main()
