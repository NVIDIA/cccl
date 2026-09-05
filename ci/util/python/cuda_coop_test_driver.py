#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run one cuda.coop CI stage with explicit provenance and test ownership."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Batch:
    name: str
    selectors: tuple[str, ...]
    workers: int
    needs_gpu: bool = False
    required: bool = True
    forbid_skips: bool = True
    selection_args: tuple[str, ...] = ()
    pytest_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage:
    modules: tuple[str, ...]
    distributions: tuple[str, ...]
    batches: tuple[Batch, ...]
    required_environment: tuple[str, ...] = ()
    required_public_symbols: tuple[tuple[str, tuple[str, ...]], ...] = ()


_COOP_TESTS = "python/cuda_coop/tests"
_NUMBA_MLIR_HOST_COMPILE_DIAGNOSTICS = tuple(
    f"{_COOP_TESTS}/backends/numba_mlir/compile/{name}"
    for name in (
        "test_common_root.py",
        "test_factories.py",
        "test_group_hierarchy.py",
        "test_nvrtc_dump.py",
    )
)
_CUTLASS_EXAMPLES_INSTALLED_ENV = "CUDA_COOP_EXAMPLES_USE_INSTALLED_CUDA_COOP"
_CUTLASS_EXAMPLE_INSTALLED_STAGES = frozenset(
    {
        "cutlass",
        "cutlass-host",
        "cutlass-final-link-qualification",
        "cutlass-cluster-qualification",
        "cutlass-sm100-qualification",
    }
)
_COMMON_INSTALLED_MODULES = ("cuda.coop",)
_COMMON_INSTALLED_DISTRIBUTIONS = ("cuda-coop",)
_CUTLASS_INSTALLED_MODULES = (
    *_COMMON_INSTALLED_MODULES,
    "cuda.coop.cutlass",
    "cutlass",
    "torch",
)
_CUTLASS_INSTALLED_DISTRIBUTIONS = (
    *_COMMON_INSTALLED_DISTRIBUTIONS,
    "nvidia-cutlass-dsl",
    "torch",
)
_MIXED_INSTALLED_MODULES = (
    *_CUTLASS_INSTALLED_MODULES,
    "cuda.coop.numba_mlir",
    "numba_cuda_mlir",
    "numba_cuda_mlir.cuda",
)
_MIXED_INSTALLED_DISTRIBUTIONS = (
    *_CUTLASS_INSTALLED_DISTRIBUTIONS,
    "numba-cuda-mlir",
)
_CUTLASS_ROOT_REQUIRED_PUBLIC_SYMBOLS = (
    ("cuda.coop", ("ThreadData", "this_block", "reduce")),
)
_NUMBA_MLIR_REQUIRED_PUBLIC_SYMBOLS = (
    (
        "numba_cuda_mlir.extending",
        (
            "WholeFunctionPlanner",
            "register_planner",
            "require_launch_config",
        ),
    ),
)
_MIXED_REQUIRED_PUBLIC_SYMBOLS = (
    *_CUTLASS_ROOT_REQUIRED_PUBLIC_SYMBOLS,
    *_NUMBA_MLIR_REQUIRED_PUBLIC_SYMBOLS,
)

STAGES: dict[str, Stage] = {
    "contracts": Stage(
        modules=(*_COMMON_INSTALLED_MODULES, "cuda.coop._core"),
        distributions=_COMMON_INSTALLED_DISTRIBUTIONS,
        required_public_symbols=_CUTLASS_ROOT_REQUIRED_PUBLIC_SYMBOLS,
        batches=(
            Batch(
                "host",
                (
                    f"{_COOP_TESTS}/contracts",
                    f"{_COOP_TESTS}/integration/examples",
                    f"{_COOP_TESTS}/integration/packaging",
                ),
                workers=6,
                pytest_args=(
                    "--cov=cuda.coop._core",
                    "--cov-branch",
                    "--cov-report=term-missing",
                    "--cov-fail-under=87",
                ),
            ),
        ),
    ),
    "numba-mlir": Stage(
        modules=(
            *_COMMON_INSTALLED_MODULES,
            "cuda.coop.numba_mlir",
            "numba_cuda_mlir",
            "numba_cuda_mlir.cuda",
        ),
        distributions=(*_COMMON_INSTALLED_DISTRIBUTIONS, "numba-cuda-mlir"),
        required_public_symbols=_NUMBA_MLIR_REQUIRED_PUBLIC_SYMBOLS,
        batches=(
            Batch(
                "host",
                (f"{_COOP_TESTS}/backends/numba_mlir/unit",),
                workers=6,
            ),
            Batch(
                "host-compile-diagnostics",
                _NUMBA_MLIR_HOST_COMPILE_DIAGNOSTICS,
                workers=6,
            ),
            Batch(
                "gpu-compile-diagnostics",
                (f"{_COOP_TESTS}/backends/numba_mlir/compile",),
                workers=0,
                needs_gpu=True,
                forbid_skips=False,
                selection_args=("-m", "gpu"),
            ),
            Batch(
                "runtime",
                (f"{_COOP_TESTS}/backends/numba_mlir/runtime",),
                workers=0,
                needs_gpu=True,
                forbid_skips=False,
            ),
        ),
    ),
    "cutlass": Stage(
        modules=_MIXED_INSTALLED_MODULES,
        distributions=_MIXED_INSTALLED_DISTRIBUTIONS,
        required_public_symbols=_MIXED_REQUIRED_PUBLIC_SYMBOLS,
        batches=(
            Batch(
                "host-and-compile",
                (
                    f"{_COOP_TESTS}/backends/cutlass/unit",
                    f"{_COOP_TESTS}/backends/cutlass/compile",
                ),
                workers=6,
                forbid_skips=False,
            ),
            Batch(
                "provider-link",
                (f"{_COOP_TESTS}/providers/cutlass",),
                workers=0,
                needs_gpu=True,
                forbid_skips=False,
                selection_args=("-m", "not requires_sm100"),
            ),
            Batch(
                "runtime",
                (f"{_COOP_TESTS}/backends/cutlass/runtime",),
                workers=0,
                needs_gpu=True,
                forbid_skips=False,
            ),
            Batch(
                "mixed-activation",
                (f"{_COOP_TESTS}/integration/compiler",),
                workers=0,
                needs_gpu=True,
            ),
        ),
    ),
    "numba-mlir-host": Stage(
        modules=(
            *_COMMON_INSTALLED_MODULES,
            "cuda.coop.numba_mlir",
            "numba_cuda_mlir",
        ),
        distributions=(*_COMMON_INSTALLED_DISTRIBUTIONS, "numba-cuda-mlir"),
        required_public_symbols=_NUMBA_MLIR_REQUIRED_PUBLIC_SYMBOLS,
        batches=(
            Batch(
                "host",
                (f"{_COOP_TESTS}/backends/numba_mlir/unit",),
                workers=6,
            ),
            Batch(
                "host-compile-diagnostics",
                _NUMBA_MLIR_HOST_COMPILE_DIAGNOSTICS,
                workers=6,
            ),
        ),
    ),
    "cutlass-host": Stage(
        modules=_CUTLASS_INSTALLED_MODULES,
        distributions=_CUTLASS_INSTALLED_DISTRIBUTIONS,
        required_public_symbols=_CUTLASS_ROOT_REQUIRED_PUBLIC_SYMBOLS,
        batches=(
            Batch(
                "host-and-compile",
                (
                    f"{_COOP_TESTS}/backends/cutlass/unit",
                    f"{_COOP_TESTS}/backends/cutlass/compile",
                ),
                workers=6,
                selection_args=("-m", "not gpu"),
            ),
        ),
    ),
    "numba-mlir-qualification": Stage(
        modules=(
            *_COMMON_INSTALLED_MODULES,
            "cuda.coop.numba_mlir",
            "numba_cuda_mlir",
            "numba_cuda_mlir.cuda",
        ),
        distributions=(*_COMMON_INSTALLED_DISTRIBUTIONS, "numba-cuda-mlir"),
        required_public_symbols=_NUMBA_MLIR_REQUIRED_PUBLIC_SYMBOLS,
        batches=(
            Batch(
                "final-link",
                (f"{_COOP_TESTS}/providers/qualification/numba_mlir",),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
            ),
            Batch(
                "stress",
                (f"{_COOP_TESTS}/backends/numba_mlir/stress",),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
            ),
        ),
    ),
    "numba-mlir-cluster-qualification": Stage(
        modules=(
            *_COMMON_INSTALLED_MODULES,
            "cuda.coop.numba_mlir",
            "numba_cuda_mlir",
            "numba_cuda_mlir.cuda",
        ),
        distributions=(*_COMMON_INSTALLED_DISTRIBUTIONS, "numba-cuda-mlir"),
        required_public_symbols=_NUMBA_MLIR_REQUIRED_PUBLIC_SYMBOLS,
        batches=(
            Batch(
                "cluster",
                (
                    f"{_COOP_TESTS}/backends/numba_mlir/runtime/test_common_profile.py",
                    f"{_COOP_TESTS}/backends/numba_mlir/runtime/test_group_hierarchy.py",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
                selection_args=("-k", "cluster"),
            ),
        ),
    ),
    "cutlass-final-link-qualification": Stage(
        modules=_CUTLASS_INSTALLED_MODULES,
        distributions=_CUTLASS_INSTALLED_DISTRIBUTIONS,
        required_public_symbols=_CUTLASS_ROOT_REQUIRED_PUBLIC_SYMBOLS,
        batches=(
            Batch(
                "final-link",
                (f"{_COOP_TESTS}/providers/cutlass/test_ltoir_inlining.py",),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
                selection_args=("-m", "not requires_sm100"),
            ),
            Batch(
                "load-store-final-link",
                (
                    f"{_COOP_TESTS}/providers/cutlass/"
                    "test_common_root_load_store_final_link.py",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
            ),
            Batch(
                "reduce-sum-scan-final-link",
                (
                    f"{_COOP_TESTS}/providers/cutlass/"
                    "test_common_root_sum_scan_final_link.py",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
            ),
            Batch(
                "exchange-final-link",
                (
                    f"{_COOP_TESTS}/providers/cutlass/"
                    "test_common_root_exchange_final_link.py",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
            ),
            Batch(
                "adjacent-discontinuity-final-link",
                (
                    f"{_COOP_TESTS}/providers/cutlass/"
                    "test_common_root_adjacent_discontinuity_final_link.py",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
            ),
            Batch(
                "shuffle-final-link",
                (
                    f"{_COOP_TESTS}/providers/cutlass/"
                    "test_common_root_shuffle_final_link.py",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
            ),
            Batch(
                "histogram-final-link",
                (
                    f"{_COOP_TESTS}/providers/cutlass/"
                    "test_common_root_histogram_final_link.py",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
            ),
            Batch(
                "run-length-decode-final-link",
                (
                    f"{_COOP_TESTS}/providers/cutlass/"
                    "test_common_root_run_length_decode_final_link.py",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
            ),
            Batch(
                "merge-sort-final-link",
                (
                    f"{_COOP_TESTS}/providers/cutlass/"
                    "test_common_root_merge_sort_final_link.py",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
            ),
            Batch(
                "radix-sort-final-link",
                (
                    f"{_COOP_TESTS}/providers/cutlass/"
                    "test_common_root_radix_sort_final_link.py",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
            ),
            Batch(
                "radix-rank-final-link",
                (
                    f"{_COOP_TESTS}/providers/cutlass/"
                    "test_common_root_radix_rank_final_link.py",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
            ),
            Batch(
                "topk-final-link",
                (
                    f"{_COOP_TESTS}/providers/cutlass/"
                    "test_common_root_topk_final_link.py",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
            ),
        ),
    ),
    "cutlass-cluster-qualification": Stage(
        modules=_CUTLASS_INSTALLED_MODULES,
        distributions=_CUTLASS_INSTALLED_DISTRIBUTIONS,
        required_public_symbols=_CUTLASS_ROOT_REQUIRED_PUBLIC_SYMBOLS,
        batches=(
            Batch(
                "cluster",
                (
                    f"{_COOP_TESTS}/backends/cutlass/runtime/test_common_profile.py",
                    f"{_COOP_TESTS}/backends/cutlass/runtime/test_group_hierarchy.py",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=True,
                selection_args=("-k", "cluster"),
            ),
        ),
    ),
    "cutlass-sm100-qualification": Stage(
        modules=_CUTLASS_INSTALLED_MODULES,
        distributions=_CUTLASS_INSTALLED_DISTRIBUTIONS,
        required_public_symbols=_CUTLASS_ROOT_REQUIRED_PUBLIC_SYMBOLS,
        batches=(
            Batch(
                "qualification",
                (
                    f"{_COOP_TESTS}/providers/cutlass/test_ltoir_inlining.py",
                    f"{_COOP_TESTS}/providers/qualification/cutlass",
                ),
                workers=0,
                needs_gpu=True,
                forbid_skips=False,
                selection_args=("-m", "requires_sm100"),
            ),
        ),
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _contains_tests(path: Path) -> bool:
    if path.is_file():
        return path.name.startswith("test_") and path.suffix == ".py"
    return path.is_dir() and any(path.rglob("test_*.py"))


def _resolve_selectors(repo_root: Path, selectors: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for selector in selectors:
        matches = (
            sorted(repo_root.glob(selector))
            if any(character in selector for character in "*?[")
            else [repo_root / selector]
        )
        for path in matches:
            path = path.resolve()
            if path not in seen and _contains_tests(path):
                resolved.append(path)
                seen.add(path)
    return resolved


def _module_locations(module_name: str) -> list[Path]:
    module = importlib.import_module(module_name)
    locations: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        locations.append(Path(module_file).resolve())
    module_path = getattr(module, "__path__", ())
    locations.extend(Path(path).resolve() for path in module_path)
    return locations


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _print_required_public_symbols(stage: Stage) -> None:
    for module_name, required_symbols in stage.required_public_symbols:
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise RuntimeError(
                f"required public API module {module_name!r} "
                f"could not be imported: {error}"
            ) from error

        missing = tuple(name for name in required_symbols if not hasattr(module, name))
        if missing:
            raise RuntimeError(
                f"required public API module {module_name!r} "
                f"is missing symbols: {', '.join(missing)}"
            )

        print(f"public API {module_name}: {', '.join(required_symbols)}")


def _print_provenance(repo_root: Path, stage: Stage) -> None:
    prefix = Path(sys.prefix).resolve()
    print(f"python={sys.version.split()[0]} executable={sys.executable}")
    print(f"environment_prefix={prefix}")

    invalid_locations: list[str] = []
    for module_name in stage.modules:
        locations = _module_locations(module_name)
        rendered = ", ".join(str(path) for path in locations) or "<namespace>"
        print(f"module {module_name}: {rendered}")
        for location in locations:
            if _is_below(location, repo_root) or not _is_below(location, prefix):
                invalid_locations.append(f"{module_name}: {location}")

    for distribution in stage.distributions:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"required distribution {distribution!r} is not installed"
            ) from error
        print(f"distribution {distribution}={version}")

    _print_required_public_symbols(stage)

    for name in stage.required_environment:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(
                f"required backend environment variable {name!r} is not set"
            )
        path = Path(value).expanduser().resolve()
        expects_directory = name.endswith("_CUDA_PATH")
        exists = path.is_dir() if expects_directory else path.is_file()
        if not exists:
            expected_kind = "directory" if expects_directory else "file"
            raise RuntimeError(f"{name} must name an existing {expected_kind}: {path}")
        print(f"environment {name}={path}")

    if invalid_locations:
        details = "\n".join(f"  {entry}" for entry in invalid_locations)
        raise RuntimeError(
            "selected backend modules must come from the active installed "
            f"environment, not a source checkout:\n{details}"
        )


def _print_gpu_provenance() -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "this cuda.coop stage requires an available NVIDIA GPU"
        ) from error
    devices = result.stdout.strip()
    if not devices:
        raise RuntimeError("nvidia-smi reported no available NVIDIA GPU")
    print(f"gpu={devices}")


def _reject_legacy_cuda_cccl() -> None:
    try:
        version = importlib.metadata.version("cuda-cccl")
    except importlib.metadata.PackageNotFoundError:
        return
    raise RuntimeError(
        "cuda-coop CI requires an independent-wheel environment, but found "
        f"cuda-cccl=={version}"
    )


def _pytest_config(repo_root: Path) -> Path:
    return repo_root / "python" / "cuda_coop" / "pyproject.toml"


def _batch_environment(stage_name: str) -> dict[str, str]:
    """Keep CUTLASS examples on the driver's validated installed package."""

    environment = os.environ.copy()
    if stage_name in _CUTLASS_EXAMPLE_INSTALLED_STAGES:
        environment[_CUTLASS_EXAMPLES_INSTALLED_ENV] = "1"
    else:
        environment.pop(_CUTLASS_EXAMPLES_INSTALLED_ENV, None)
    return environment


def _run_batch(
    repo_root: Path,
    config: Path,
    stage_name: str,
    batch: Batch,
    selectors: Sequence[Path],
    *,
    dry_run: bool,
) -> None:
    relative_selectors = [os.fspath(path.relative_to(repo_root)) for path in selectors]
    required_backend = {
        "numba-mlir": "numba_mlir",
        "numba-mlir-host": "numba_mlir",
        "numba-mlir-qualification": "numba_mlir",
        "numba-mlir-cluster-qualification": "numba_mlir",
        "cutlass": "cutlass",
        "cutlass-host": "cutlass",
        "cutlass-final-link-qualification": "cutlass",
        "cutlass-cluster-qualification": "cutlass",
        "cutlass-sm100-qualification": "cutlass",
    }.get(stage_name)
    common = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        os.fspath(config),
        "--strict-config",
        "--strict-markers",
        *(
            ("--require-cuda-coop-backend", required_backend)
            if required_backend is not None
            else ()
        ),
        *relative_selectors,
    ]
    collect_command = [
        *common,
        *batch.selection_args,
        "--collect-only",
        "-qq",
    ]
    test_command = [
        *common,
        *(("--forbid-cuda-coop-skips",) if batch.forbid_skips else ()),
        "-n",
        str(batch.workers),
        "-v",
        "--durations=20",
        *batch.selection_args,
        *batch.pytest_args,
    ]

    print(
        f"stage={stage_name} batch={batch.name} workers={batch.workers} "
        f"gpu={str(batch.needs_gpu).lower()} "
        f"forbid_skips={str(batch.forbid_skips).lower()}"
    )
    print(f"collect command: {' '.join(collect_command)}")
    print(f"test command: {' '.join(test_command)}")
    if dry_run:
        return

    environment = _batch_environment(stage_name)

    # Running collection as a separate preflight catches import/configuration
    # failures before a long GPU batch begins. At -qq, pytest renders one
    # ``path: count`` line per collected module, which gives the CI log an
    # explicit selected count without printing thousands of node IDs.
    collection = subprocess.run(
        collect_command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    sys.stdout.write(collection.stdout)
    sys.stderr.write(collection.stderr)
    if collection.returncode != 0:
        raise RuntimeError(
            f"collection preflight failed for {stage_name}/{batch.name} "
            f"with exit code {collection.returncode}"
        )
    counts = [
        int(match.group(1))
        for line in collection.stdout.splitlines()
        if (match := re.search(r":\s+(\d+)\s*$", line))
    ]
    selected_count = sum(counts)
    if selected_count == 0:
        raise RuntimeError(
            f"collection preflight selected zero tests for {stage_name}/{batch.name}"
        )
    print(
        f"stage={stage_name} batch={batch.name} selected_tests={selected_count}",
        flush=True,
    )

    result = subprocess.run(
        test_command,
        cwd=repo_root,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pytest failed for {stage_name}/{batch.name} "
            f"with exit code {result.returncode}"
        )


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    stage = STAGES[args.stage]

    resolved_batches: list[tuple[Batch, list[Path]]] = []
    missing_batches: list[Batch] = []
    for batch in stage.batches:
        selectors = _resolve_selectors(repo_root, batch.selectors)
        if selectors:
            resolved_batches.append((batch, selectors))
        elif batch.required:
            missing_batches.append(batch)

    if missing_batches:
        configured = "\n".join(
            f"  {batch.name}: {', '.join(batch.selectors)}" for batch in missing_batches
        )
        raise RuntimeError(
            f"stage {args.stage!r} is missing required test batches:\n{configured}"
        )

    if not args.dry_run:
        _reject_legacy_cuda_cccl()
        _print_provenance(repo_root, stage)
        if any(batch.needs_gpu for batch, _ in resolved_batches):
            _print_gpu_provenance()

    for batch, selectors in resolved_batches:
        _run_batch(
            repo_root,
            _pytest_config(repo_root),
            args.stage,
            batch,
            selectors,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
