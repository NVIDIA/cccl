# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Render the human-readable cuda.coop coverage table from its test manifest."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from tests.support.coverage import CoverageManifest, load_coverage_manifest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PACKAGE_ROOT / "tests" / "contracts" / "coverage.toml"
DOCUMENT = (
    PACKAGE_ROOT
    / "docs"
    / "fern"
    / "fern"
    / "docs"
    / "pages"
    / "api"
    / "coverage-matrix.mdx"
)
START = "{/* BEGIN GENERATED TEST COVERAGE */}"
END = "{/* END GENERATED TEST COVERAGE */}"
STATUS_ORDER = (
    "signature_only",
    "placeholder",
    "native_equivalent",
    "blocked",
    "unsupported",
    "not_applicable",
)
_PRIVATE_QUALIFIED_SURFACES = frozenset(
    {
        "cuda.coop.cutlass._block",
        "cuda.coop.cutlass._warp",
        "cuda.coop.numba_mlir._block",
        "cuda.coop.numba_mlir._warp",
    }
)


@dataclass
class _Surface:
    backend: str
    scope: str
    value_model: str
    provider: str
    operations_by_status: dict[str, list[str]]
    executable_by_enforcement: dict[str, list[str]]


def _scope_and_operation(scenario: str) -> tuple[str, str]:
    scope, operation = scenario.rsplit(".", 1)
    return scope, operation


def _surfaces(manifest: CoverageManifest) -> list[_Surface]:
    grouped: dict[tuple[str, ...], _Surface] = {}
    for capability in manifest.capabilities:
        # Keep internal scoped adapters in the executable coverage authority
        # while omitting them from the public Fern matrix.
        if capability.public_surface in _PRIVATE_QUALIFIED_SURFACES:
            continue
        scope, operation = _scope_and_operation(capability.scenario)
        key = (
            capability.backend,
            scope,
            capability.public_surface,
            capability.value_model,
            capability.provider,
        )
        surface = grouped.setdefault(
            key,
            _Surface(
                backend=capability.backend,
                scope=scope,
                value_model=capability.value_model,
                provider=capability.provider,
                operations_by_status={status: [] for status in STATUS_ORDER},
                executable_by_enforcement={
                    enforcement: [] for enforcement in ("required", "migration")
                },
            ),
        )
        if capability.status == "executable":
            surface.executable_by_enforcement[capability.enforcement].append(operation)
        else:
            surface.operations_by_status[capability.status].append(operation)
    return list(grouped.values())


def render(manifest: CoverageManifest) -> str:
    lines = [
        START,
        "",
        "> This table is generated from the common V1 matrix and "
        "`tests/contracts/coverage.toml`.",
        "> Edit the relevant authority, then run "
        "`python -m tools.render_test_coverage`.",
        "",
        "| Backend | Scope | Value model | Direct evidence required | Migration fallback | Other declared states | Provider route |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for surface in _surfaces(manifest):
        required = ", ".join(
            f"`{name}`" for name in surface.executable_by_enforcement["required"]
        )
        migration = ", ".join(
            f"`{name}`" for name in surface.executable_by_enforcement["migration"]
        )
        declared: list[str] = []
        for status in STATUS_ORDER:
            names = surface.operations_by_status[status]
            if names:
                rendered = ", ".join(f"`{name}`" for name in names)
                declared.append(f"{status.replace('_', ' ')}: {rendered}")
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{surface.backend}`",
                    f"`{surface.scope}`",
                    f"`{surface.value_model}`",
                    required or "-",
                    migration or "-",
                    "<br />".join(declared) or "-",
                    surface.provider.replace("|", "\\|"),
                )
            )
            + " |"
        )
    lines.extend(("", END))
    return "\n".join(lines)


def update(document: str, generated: str) -> str:
    if document.count(START) != 1 or document.count(END) != 1:
        raise ValueError(
            f"{DOCUMENT} must contain exactly one generated coverage block"
        )
    before, remainder = document.split(START, 1)
    _, after = remainder.split(END, 1)
    return before + generated + after


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of rewriting when the checked-in table is stale",
    )
    args = parser.parse_args()

    manifest = load_coverage_manifest(MANIFEST)
    current = DOCUMENT.read_text(encoding="utf-8")
    expected = update(current, render(manifest))
    if current == expected:
        return 0
    if args.check:
        print(
            f"{DOCUMENT.relative_to(PACKAGE_ROOT)} is stale; "
            "run python -m tools.render_test_coverage",
            file=sys.stderr,
        )
        return 1
    DOCUMENT.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
