# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared pytest policy and fixtures for cuda.coop."""

from __future__ import annotations

import hashlib
import os
import random
import sys
from collections.abc import Callable, Iterable
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# Keep unrelated in-process tests independent of whichever optional DSL
# runtimes happen to be installed on the test host. Dedicated subprocess
# contracts remove this setting when exercising default-on registration.
os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"

from .support.coverage import (
    EVIDENCE_PROPERTY,
    CoverageManifest,
    EvidenceRun,
    audit_coverage_claims,
    audit_matrix_sizes,
    claims_from_items,
    load_coverage_manifest,
    required_collection_path_errors,
)
from .support.paths import PACKAGE_ROOT, TESTS_ROOT

_BACKEND_MODULES = {
    "cutlass": "cuda.coop.cutlass",
    "numba_mlir": "cuda.coop.numba_mlir",
}
_BACKEND_MARKERS = {
    "cutlass": ("backend_cutlass",),
    "numba_mlir": ("backend_numba_mlir",),
}
_PROVIDER_MARKERS = {
    "cutlass": ("backend_cutlass",),
    "numba_mlir": ("backend_numba_mlir",),
}
_CUTLASS_PROVIDER_ENV = {
    "CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT": "ltoir",
}
_COVERAGE_MANIFEST = TESTS_ROOT / "contracts" / "coverage.toml"
_EVIDENCE_COLLECTED_OUTPUT = "cuda_coop_evidence_collected"
_EVIDENCE_COLLECTION_ERRORS_OUTPUT = "cuda_coop_evidence_collection_errors"
_EVIDENCE_EXPECTED_OUTPUT = "cuda_coop_evidence_expected"
_EVIDENCE_SELECTED_CELLS_OUTPUT = "cuda_coop_evidence_selected_cells"
_EVIDENCE_SELECTED_NODEIDS_OUTPUT = "cuda_coop_evidence_selected_nodeids"


class _EvidenceOutcomePlugin:
    def __init__(self) -> None:
        self.run = EvidenceRun()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.run.record_report(report)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        self.run.record_collected_nodeid(report.nodeid)


def pytest_configure(config: pytest.Config) -> None:
    # Test modules import the in-repo ``examples``/``benchmarks`` packages at
    # module scope. Appending (never prepending) the package root keeps those
    # importable from any invocation directory while the installed ``cuda``
    # namespace still resolves ahead of the source checkout.
    package_root = str(PACKAGE_ROOT)
    if package_root not in sys.path:
        sys.path.append(package_root)
    tracker = _EvidenceOutcomePlugin()
    config._cuda_coop_evidence_tracker = tracker
    config.pluginmanager.register(tracker, "cuda-coop-evidence-outcomes")


def _publish_worker_evidence(config: pytest.Config) -> None:
    """Publish a worker inventory even when collection creates no test items."""

    if not hasattr(config, "workeroutput"):
        return
    tracker = config._cuda_coop_evidence_tracker
    config.workeroutput[_EVIDENCE_COLLECTED_OUTPUT] = tracker.run.collected_nodeids
    config.workeroutput[_EVIDENCE_EXPECTED_OUTPUT] = tracker.run.expected_payload()
    config.workeroutput[_EVIDENCE_SELECTED_CELLS_OUTPUT] = tracker.run.selected_cells
    config.workeroutput[_EVIDENCE_SELECTED_NODEIDS_OUTPUT] = (
        tracker.run.selected_nodeids
    )
    config.workeroutput.setdefault(_EVIDENCE_COLLECTION_ERRORS_OUTPUT, ())


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node: Any, ids: Iterable[str]) -> None:
    """Preserve successful worker collection paths on the controller."""

    tracker = node.config._cuda_coop_evidence_tracker
    for nodeid in ids:
        tracker.run.record_selected_nodeid(nodeid)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: object | None) -> None:
    """Merge a completed worker's selected evidence inventory."""

    if error is not None:
        return
    output = getattr(node, "workeroutput", {})
    tracker = node.config._cuda_coop_evidence_tracker
    worker = str(node.workerinput.get("workerid", "unknown worker"))
    tracker.run.merge_worker_inventory(
        worker,
        expected=output.get(_EVIDENCE_EXPECTED_OUTPUT),
        selected_cells=output.get(_EVIDENCE_SELECTED_CELLS_OUTPUT),
        collected_nodeids=output.get(_EVIDENCE_COLLECTED_OUTPUT),
        selected_nodeids=output.get(_EVIDENCE_SELECTED_NODEIDS_OUTPUT),
        collection_errors=output.get(_EVIDENCE_COLLECTION_ERRORS_OUTPUT),
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("cuda.coop")
    group.addoption(
        "--coop-random-seed",
        action="store",
        default="0",
        metavar="INTEGER",
        help="base seed mixed with each test node ID (default: 0)",
    )
    group.addoption(
        "--require-cuda-coop-backend",
        action="append",
        choices=sorted(_BACKEND_MODULES),
        default=[],
        metavar="BACKEND",
        help="turn an unavailable optional backend into a test failure",
    )
    group.addoption(
        "--coop-coverage-report",
        action="store_true",
        default=False,
        help="report conformance-evidence migration gaps after the test run",
    )
    group.addoption(
        "--forbid-cuda-coop-skips",
        action="store_true",
        default=False,
        help="fail the selected cuda.coop lane if any test is skipped",
    )


def _base_seed(config: pytest.Config) -> int:
    value = config.getoption("--coop-random-seed")
    try:
        return int(value, 0)
    except ValueError as exc:
        raise pytest.UsageError(
            f"--coop-random-seed must be an integer, got {value!r}"
        ) from exc


def _node_seed(nodeid: str, base_seed: int) -> int:
    digest = hashlib.blake2b(nodeid.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") ^ base_seed


def _explicit_collection_arguments(
    config: pytest.Config,
) -> tuple[tuple[Path, bool], ...]:
    invocation_dir = Path(config.invocation_params.dir)
    selections: list[tuple[Path, bool]] = []
    for argument in config.args:
        path_text, separator, _ = str(argument).partition("::")
        path = Path(path_text)
        if not path.is_absolute():
            path = invocation_dir / path
        selections.append((path.resolve(), bool(separator)))
    return tuple(selections)


def _explicit_collection_paths(config: pytest.Config) -> tuple[Path, ...]:
    return tuple(path for path, _ in _explicit_collection_arguments(config))


def _explicit_node_collection_paths(config: pytest.Config) -> frozenset[Path]:
    return frozenset(
        path
        for path, selects_node in _explicit_collection_arguments(config)
        if selects_node
    )


def _explicit_full_collection_roots(config: pytest.Config) -> tuple[Path, ...]:
    return tuple(
        path
        for path, selects_node in _explicit_collection_arguments(config)
        if not selects_node
    )


def _is_exact_only_path(config: pytest.Config, path: Path) -> bool:
    path = path.resolve()
    return path in _explicit_node_collection_paths(config) and not any(
        root == path or root in path.parents
        for root in _explicit_full_collection_roots(config)
    )


def _coverage_selected_file_paths(
    config: pytest.Config, paths: Iterable[Path]
) -> tuple[str, ...]:
    selected_paths: list[str] = []
    for path in paths:
        path = path.resolve()
        if _is_exact_only_path(config, path):
            continue
        try:
            relative = path.relative_to(PACKAGE_ROOT)
        except ValueError:
            continue
        selected_paths.append(relative.as_posix())
    return tuple(dict.fromkeys(selected_paths))


def _paths_from_nodeids(
    config: pytest.Config, nodeids: Iterable[str]
) -> tuple[Path, ...]:
    root_path = Path(config.rootpath)
    paths: list[Path] = []
    for nodeid in nodeids:
        path = Path(nodeid.split("::", 1)[0])
        if not path.is_absolute():
            path = root_path / path
        paths.append(path.resolve())
    return tuple(paths)


def _coverage_observed_paths(
    config: pytest.Config,
    nodeids: Iterable[str],
    selected_nodeids: Iterable[str],
) -> tuple[str, ...]:
    """Return complete lanes represented by collection or execution reports."""

    observed_paths = _paths_from_nodeids(config, nodeids)
    selected_paths = frozenset(_paths_from_nodeids(config, selected_nodeids))
    paths = list(_coverage_selected_file_paths(config, observed_paths))
    for exact_path in _explicit_node_collection_paths(config):
        if exact_path not in observed_paths or exact_path in selected_paths:
            continue
        try:
            paths.append(exact_path.relative_to(PACKAGE_ROOT).as_posix())
        except ValueError:
            continue
    return tuple(dict.fromkeys(paths))


def _is_explicit_subtree(root: Path, config: pytest.Config) -> bool:
    return any(
        selected == root or root in selected.parents
        for selected in _explicit_collection_paths(config)
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Keep optional backend imports out of an unqualified root collection."""

    path = Path(collection_path).resolve()
    optional_roots = (
        *(TESTS_ROOT / "backends" / backend for backend in _BACKEND_MARKERS),
        *(TESTS_ROOT / "providers" / provider for provider in _PROVIDER_MARKERS),
        TESTS_ROOT / "providers" / "qualification",
        TESTS_ROOT / "integration" / "compiler",
    )
    required_backends = set(config.getoption("--require-cuda-coop-backend"))
    for root in optional_roots:
        root = root.resolve()
        if path != root and root not in path.parents:
            continue
        if _is_explicit_subtree(root, config):
            return None
        if root.parent.name == "backends" and root.name in required_backends:
            return None
        return True
    return None


def _automatic_markers(item: pytest.Item) -> tuple[str, ...]:
    try:
        relative = Path(item.path).resolve().relative_to(TESTS_ROOT)
    except ValueError:
        return ()

    parts = relative.parts
    if not parts:
        return ()

    markers: list[str] = []
    if parts[0] == "contracts":
        markers.extend(("backend_core", "contract"))
    elif parts[0] == "backends" and len(parts) > 2:
        markers.extend(_BACKEND_MARKERS.get(parts[1], ()))
        layer = parts[2]
        if layer == "unit":
            markers.append("unit")
        elif layer == "compile":
            markers.append("compile")
        elif layer == "runtime":
            markers.extend(("runtime", "gpu"))
        elif layer == "stress":
            markers.extend(("stress", "qualification", "gpu"))
        elif layer == "link":
            markers.append("link")
        elif layer == "qualification":
            markers.extend(("qualification", "gpu"))
    elif parts[0] == "providers" and len(parts) > 1:
        provider = (
            parts[2] if parts[1] == "qualification" and len(parts) > 2 else parts[1]
        )
        markers.extend(_PROVIDER_MARKERS.get(provider, ()))
        if "link" in parts:
            markers.append("link")
        if "qualification" in parts:
            markers.extend(("qualification", "gpu"))
    elif parts[0] == "integration" and len(parts) > 1:
        integration_marker = {
            "compiler": "compile",
            "docs": "docs",
            "examples": "examples",
            "packaging": "packaging",
        }.get(parts[1])
        if integration_marker is not None:
            markers.append(integration_marker)
    return tuple(dict.fromkeys(markers))


@pytest.fixture
def deterministic_seed(request: pytest.FixtureRequest) -> int:
    """Return a stable seed derived from the logical test ID."""

    seed = _node_seed(request.node.nodeid, _base_seed(request.config))
    request.node.user_properties.append(("cuda_coop_seed", seed))
    return seed


@pytest.fixture
def python_rng(deterministic_seed: int) -> random.Random:
    """Return an isolated deterministic Python random-number generator."""

    return random.Random(deterministic_seed)


@pytest.fixture(autouse=True)
def deterministic_rng(deterministic_seed: int):
    """Seed and restore legacy RNGs, and provide an isolated NumPy generator."""

    import numpy as np

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(deterministic_seed)
    np.random.seed(deterministic_seed % (2**32))
    try:
        yield np.random.default_rng(deterministic_seed)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@pytest.fixture(autouse=True)
def isolated_qualified_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep process-wide qualified-import activation isolated between tests."""

    root_api = import_module("cuda.coop._core.root_api")
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)


@pytest.fixture
def qualified_cutlass_backend(isolated_qualified_backend: None) -> None:
    """Restore CUTLASS import activation after pytest collection isolation."""

    del isolated_qualified_backend
    root_api = import_module("cuda.coop._core.root_api")
    root_api._register_qualified_backend("cuda.coop.cutlass")


@pytest.fixture
def numpy_rng(deterministic_rng: Any):
    """Compatibility name for the isolated deterministic NumPy generator."""

    return deterministic_rng


@pytest.fixture
def scoped_env(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., None]:
    """Return a helper that restores every environment update after the test."""

    def update(**values: str | os.PathLike[str] | None) -> None:
        for name, value in values.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, os.fspath(value))

    return update


@pytest.fixture
def failure_artifact_dir(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Return a pytest-retained directory and report it when the test fails."""

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    paths = getattr(request.node, "_cuda_coop_artifact_paths", [])
    paths.append(artifact_dir)
    request.node._cuda_coop_artifact_paths = paths
    return artifact_dir


@pytest.fixture
def compiler_cache_dir(
    failure_artifact_dir: Path, scoped_env: Callable[..., None]
) -> Path:
    """Give compiler-backed tests isolated, automatically restored caches."""

    cache_dir = failure_artifact_dir / "compiler-cache"
    cache_dir.mkdir()
    scoped_env(
        CUDA_CACHE_PATH=cache_dir / "cuda",
        CUPY_CACHE_DIR=cache_dir / "cupy",
        NUMBA_CACHE_DIR=cache_dir / "numba",
    )
    return cache_dir


@pytest.fixture
def source_examples(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Expose repository examples for one test and restore import state after it."""

    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    return PACKAGE_ROOT / "examples"


@pytest.fixture
def register_cutlass_trace_finalize_hook():
    """Register a CUTLASS trace hook through its supported DSL seam."""

    base_dsl = import_module("cutlass.base_dsl.dsl").BaseDSL._get_dsl()
    original_hooks = list(base_dsl._trace_finalize_hooks)

    def register(hook: Callable[..., None]) -> None:
        base_dsl.register_trace_finalize_hook(hook)

    try:
        yield register
    finally:
        base_dsl._trace_finalize_hooks = original_hooks


@pytest.fixture(scope="session")
def cutlass_provider_environment(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Isolate CUTLASS provider artifacts and restore process state at session end."""

    cache_dir = tmp_path_factory.mktemp("cuda-coop-cutlass-provider-cache")
    values = {
        **_CUTLASS_PROVIDER_ENV,
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR": os.fspath(cache_dir),
    }
    original = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield cache_dir
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture
def optional_backend(
    request: pytest.FixtureRequest,
) -> Callable[[str], ModuleType]:
    """Import an optional backend inside a test, skipping only local soft probes."""

    required = set(request.config.getoption("--require-cuda-coop-backend"))

    def load(backend: str) -> ModuleType:
        try:
            module_name = _BACKEND_MODULES[backend]
        except KeyError as exc:
            raise ValueError(f"unknown cuda.coop backend {backend!r}") from exc
        try:
            return import_module(module_name)
        except ImportError as exc:
            message = f"{backend} backend is unavailable: {exc}"
            if backend in required:
                pytest.fail(message, pytrace=False)
            pytest.skip(message)

    return load


@pytest.fixture(scope="session")
def backend_prerequisite(
    pytestconfig: pytest.Config,
) -> Callable[[str, bool, str], None]:
    """Skip unavailable optional prerequisites, or fail a required backend lane."""

    required = set(pytestconfig.getoption("--require-cuda-coop-backend"))

    def check(backend: str, available: bool, reason: str) -> None:
        if available:
            return
        if backend in required:
            pytest.fail(
                f"required cuda.coop backend {backend!r}: {reason}",
                pytrace=False,
            )
        pytest.skip(reason)

    return check


@pytest.fixture
def synchronize_backend() -> Callable[[str], None]:
    """Return an explicit backend synchronizer; importing it remains opt-in."""

    def synchronize(backend: str) -> None:
        if backend == "numba_mlir":
            cuda = import_module("numba_cuda_mlir.cuda")
            cuda.synchronize()
            return
        if backend == "cutlass":
            torch = import_module("torch")
            torch.cuda.synchronize()
            return
        raise ValueError(f"unknown cuda.coop backend {backend!r}")

    return synchronize


@pytest.fixture(scope="session")
def coverage_manifest() -> CoverageManifest:
    """Load the test-side capability contract without importing a backend."""

    return load_coverage_manifest(_COVERAGE_MANIFEST)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    outcome = yield
    report = outcome.get_result()
    if not report.failed:
        return
    seed = dict(item.user_properties).get("cuda_coop_seed")
    if seed is None:
        seed = _node_seed(item.nodeid, _base_seed(item.config))
    report.sections.append(("cuda.coop deterministic seed", str(seed)))
    artifact_paths = list(getattr(item, "_cuda_coop_artifact_paths", ()))
    tmp_path = item.funcargs.get("tmp_path")
    if isinstance(tmp_path, Path):
        artifact_paths.append(tmp_path)
    provider_cache = os.environ.get("CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR")
    if provider_cache:
        artifact_paths.append(Path(provider_cache))
    if artifact_paths:
        retained = "\n".join(os.fspath(path) for path in dict.fromkeys(artifact_paths))
        report.sections.append(("cuda.coop retained artifacts", retained))


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> Any:
    del session
    for item in items:
        for marker in _automatic_markers(item):
            item.add_marker(getattr(pytest.mark, marker))
        try:
            relative = Path(item.path).resolve().relative_to(TESTS_ROOT)
        except ValueError:
            continue
        if relative.parts[:2] in {
            ("backends", "cutlass"),
            ("providers", "cutlass"),
        } or relative.parts[:3] == ("providers", "qualification", "cutlass"):
            item.add_marker(pytest.mark.usefixtures("cutlass_provider_environment"))

    yield

    manifest = load_coverage_manifest(_COVERAGE_MANIFEST)
    claims, marker_errors = claims_from_items(items)
    tracker = config._cuda_coop_evidence_tracker
    for item in items:
        tracker.run.record_selected_nodeid(item.nodeid)
        item_claims, _ = claims_from_items((item,))
        if item_claims:
            tracker.run.record_expected(item.nodeid, item_claims)
        if _is_exact_only_path(config, Path(item.path)):
            tracker.run.record_selected_cells(claim.cell for claim in item_claims)
        for claim in item_claims:
            item.user_properties.append(
                (
                    EVIDENCE_PROPERTY,
                    (
                        claim.scenario,
                        claim.backend,
                        claim.evidence,
                        claim.source,
                    ),
                )
            )
    _publish_worker_evidence(config)
    matrix_errors = audit_matrix_sizes(manifest, items)
    collection_path_errors = required_collection_path_errors(
        manifest, package_root=PACKAGE_ROOT
    )
    audit = audit_coverage_claims(
        manifest,
        claims,
        selected_paths=_coverage_observed_paths(
            config,
            tracker.run.nodeids,
            tracker.run.selected_nodeids,
        ),
        selected_cells=tracker.run.selected_cells,
    )
    errors = (
        *marker_errors,
        *matrix_errors,
        *collection_path_errors,
        *audit.errors,
    )
    if errors:
        if hasattr(config, "workeroutput"):
            config.workeroutput[_EVIDENCE_COLLECTION_ERRORS_OUTPUT] = errors
            config._cuda_coop_coverage_audit = audit
            items.clear()
            return
        details = "\n".join(f"- {error}" for error in errors)
        raise pytest.UsageError(
            f"cuda.coop conformance-evidence audit failed:\n{details}"
        )
    if hasattr(config, "workeroutput"):
        config.workeroutput[_EVIDENCE_COLLECTION_ERRORS_OUTPUT] = ()
    config._cuda_coop_coverage_audit = audit


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: pytest.Config
) -> None:
    del exitstatus
    if not config.getoption("--coop-coverage-report"):
        return
    audit = getattr(config, "_cuda_coop_coverage_audit", None)
    if audit is None:
        return
    terminalreporter.section("cuda.coop qualified-surface evidence migration")
    terminalreporter.write_line(
        f"{len(audit.migration_gaps)} declared evidence cells on migration "
        "surfaces still lack direct marker ownership"
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Do not let a required cuda.coop lane hide missing prerequisites."""

    del exitstatus
    config = session.config
    if hasattr(config, "workerinput"):
        _publish_worker_evidence(config)
        return
    if config.option.collectonly:
        return

    tracker = config._cuda_coop_evidence_tracker
    manifest = load_coverage_manifest(_COVERAGE_MANIFEST)
    selected_paths = _coverage_observed_paths(
        config,
        tracker.run.nodeids,
        tracker.run.selected_nodeids,
    )
    audit = audit_coverage_claims(
        manifest,
        tracker.run.earned_claims(),
        selected_paths=selected_paths,
        selected_cells=tracker.run.selected_cells,
    )
    config._cuda_coop_coverage_audit = audit
    terminalreporter = config.pluginmanager.get_plugin("terminalreporter")
    evidence_errors = tuple(
        dict.fromkeys((*tracker.run.metadata_errors, *audit.errors))
    )
    if evidence_errors:
        if terminalreporter is not None:
            terminalreporter.write_line(
                "ERROR: cuda.coop tests did not earn required conformance evidence:",
                red=True,
                bold=True,
            )
            for error in evidence_errors:
                terminalreporter.write_line(f"- {error}", red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED

    if terminalreporter is None:
        return

    skipped_reports = terminalreporter.stats.get("skipped", ())
    if config.getoption("--forbid-cuda-coop-skips") and skipped_reports:
        terminalreporter.write_line(
            "ERROR: required cuda.coop lane skipped "
            f"{len(skipped_reports)} selected test(s); prerequisites must fail "
            "the lane instead of being silently omitted",
            red=True,
            bold=True,
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED

    required = config.getoption("--require-cuda-coop-backend")
    if required and not terminalreporter.stats.get("passed"):
        terminalreporter.write_line(
            "ERROR: required cuda.coop backend lane produced no passing tests "
            f"(required={','.join(required)}, skipped={len(skipped_reports)})",
            red=True,
            bold=True,
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
