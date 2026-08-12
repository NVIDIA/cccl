#!/usr/bin/env python3
"""Parse the `sass:` section of ci/matrix.yaml into a GitHub Actions matrix.

The section selects the build configurations that the SASS comparison uses. The
configuration is kept in the matrix file, so that it can be trimmed without a
change to any script.
"""

import argparse
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class Workflow(StrEnum):
    """The CI workflows that ci/matrix.yaml declares jobs for."""

    PULL_REQUEST = "pull_request"
    NIGHTLY = "nightly"
    WEEKLY = "weekly"


def resolve_ctk(matrix: dict[str, Any], ctk: str) -> str:
    """Resolve a `ctk:` value to the version that `launch.sh --cuda` wants.

    `ctk_versions` names the real versions and lists the aliases of each, so
    `13.X` follows the newest CTK.
    """
    versions = matrix["ctk_versions"]
    for version, spec in versions.items():
        if str(version) == ctk or ctk in spec.get("alias", []):
            return str(version)
    raise KeyError(f"Unknown ctk '{ctk}'. Valid: {sorted(map(str, versions))}")


def resolve_cxx(matrix: dict[str, Any], cxx: str) -> str:
    """Resolve a `cxx:` value to what `launch.sh --host` wants.

    The container tag is not always the compiler name: clang lives in the `llvm`
    images. `host_compilers` holds the mapping.
    """
    id = cxx.rstrip("0123456789.")
    version = cxx[len(id) :]
    compilers = matrix["host_compilers"]
    return compilers[id]["container_tag"] + version


def matrix_entry(matrix: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Convert one `sass:` entry to a matrix row."""
    return {
        "id": config["id"],
        "gpu": config["gpu"],
        "launch_args": (
            f"--cuda {resolve_ctk(matrix, config['ctk'])} "
            f"--host {resolve_cxx(matrix, config['cxx'])} "
        ),
        "preset": config["preset"],
        "archs": config["archs"],
        # The filters reach the workflow as JSON, because a GitHub Actions
        # matrix value cannot hold a list.
        "target_filters_json": json.dumps(
            config["target_filters"], separators=(",", ":")
        ),
    }


def parse_matrix(path: Path, workflow: Workflow) -> dict[str, Any]:
    with path.open() as f:
        matrix = yaml.safe_load(f)

    configs = matrix["sass"][workflow]
    return {"include": [matrix_entry(matrix, config) for config in configs]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse ci/matrix.yaml sass entries for GitHub Actions."
    )
    parser.add_argument(
        "matrix_yaml",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "matrix.yaml",
        help="Path to ci/matrix.yaml. Default: the one in this checkout.",
    )
    parser.add_argument(
        "--workflow",
        type=Workflow,
        choices=list(Workflow),
        default=Workflow.PULL_REQUEST,
        help="Which `sass:` subsection to read. Default: %(default)s.",
    )
    args = parser.parse_args()

    print(json.dumps(parse_matrix(args.matrix_yaml, args.workflow)))


if __name__ == "__main__":
    main()
