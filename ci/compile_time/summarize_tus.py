#!/usr/bin/env python3

import argparse
import csv
import subprocess
from pathlib import Path

GENERATED_TU_MARKER = "/headers/"
GENERATED_TU_SOURCE_SUFFIXES = (".cu", ".cpp", ".cxx", ".cc", ".c")
PREPROCESSED_TU_SUFFIX = ".cpp4.ii"


def strip_generated_tu_suffix(path_text: str) -> str:
    for suffix in GENERATED_TU_SOURCE_SUFFIXES:
        if path_text.endswith(suffix):
            return path_text[: -len(suffix)]
    return path_text


def generated_tu_input(tu: Path) -> str:
    parts = tu.as_posix().split(GENERATED_TU_MARKER, 1)
    if len(parts) != 2:
        return tu.as_posix()

    rel = parts[1].split("/", 1)
    if len(rel) != 2:
        return tu.as_posix()

    return strip_generated_tu_suffix(rel[1])


def find_preprocessed_tus(build_dir: Path) -> list[Path]:
    return sorted(build_dir.glob("**/headers/**/*.cpp4.ii"))


def tu_source_for_preprocessed_tu(pp_path: Path) -> Path:
    pp_text = pp_path.as_posix()
    if pp_text.endswith(PREPROCESSED_TU_SUFFIX):
        return Path(pp_text[: -len(PREPROCESSED_TU_SUFFIX)])
    return pp_path.with_suffix("")


def run_cloc(preprocessed_tus: list[Path], processes: int) -> dict[str, int]:
    if not preprocessed_tus:
        return {}

    command = [
        "cloc",
        "--csv",
        "--by-file",
        "--skip-uniqueness",
        "--processes",
        str(processes),
        "--force-lang=C++,ii",
        *[path.as_posix() for path in preprocessed_tus],
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    loc_by_file: dict[str, int] = {}
    reader = csv.reader(result.stdout.splitlines())
    for row in reader:
        if len(row) < 5 or row[1] == "filename":
            continue
        try:
            loc_by_file[row[1]] = int(row[4])
        except ValueError:
            continue
    return loc_by_file


def write_summary(
    output_csv: Path,
    preprocessed_tus: list[Path],
    loc_by_file: dict[str, int],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tu_input",
                "transitive_loc",
                "tu_source",
                "preprocessed_tu",
            ],
        )
        writer.writeheader()
        for pp_path in preprocessed_tus:
            tu_path = tu_source_for_preprocessed_tu(pp_path)
            writer.writerow(
                {
                    "tu_input": generated_tu_input(tu_path),
                    "transitive_loc": loc_by_file.get(pp_path.as_posix(), 0),
                    "tu_source": tu_path.as_posix(),
                    "preprocessed_tu": pp_path.as_posix(),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize generated TU inputs and preprocessed LOC."
    )
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--cloc-processes",
        type=int,
        default=0,
        help="cloc process count; 0 uses nproc --all --ignore=2 when available",
    )
    args = parser.parse_args()

    build_dir = args.build_dir.resolve(strict=False)
    preprocessed_tus = find_preprocessed_tus(build_dir)
    if not preprocessed_tus:
        raise SystemExit(f"no preprocessed generated TUs found under {build_dir}")

    processes = args.cloc_processes
    if processes <= 0:
        try:
            processes = int(
                subprocess.check_output(
                    ["nproc", "--all", "--ignore=2"], text=True
                ).strip()
            )
        except (subprocess.SubprocessError, ValueError):
            processes = 1

    write_summary(
        output_csv=args.output_csv,
        preprocessed_tus=preprocessed_tus,
        loc_by_file=run_cloc(preprocessed_tus, processes),
    )
    print(f"wrote {len(preprocessed_tus)} generated TU row(s) to {args.output_csv}")


if __name__ == "__main__":
    main()
