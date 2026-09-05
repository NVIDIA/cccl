# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ast
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from cuda.coop._core import root_api

from ...support import coverage as coverage_support
from ...support.cases.api_contracts import (
    COMMON_PROFILE_MATRIX,
)
from ...support.coverage import (
    EVIDENCE_PROPERTY,
    CoverageClaim,
    CoverageManifest,
    CoverageManifestError,
    EvidenceRun,
    audit_coverage_claims,
    audit_matrix_sizes,
    claims_from_items,
    load_coverage_manifest,
    migration_source_errors,
    required_collection_path_errors,
)
from ...support.paths import PACKAGE_ROOT, REPO_ROOT, TESTS_ROOT

_GROUP_FIRST_SURFACES = {
    "core": "cuda.coop",
    "cutlass": "cuda.coop.cutlass",
    "numba_mlir": "cuda.coop.numba_mlir",
}


def _literal_exports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in statement.targets
        ):
            continue
        exports = ast.literal_eval(statement.value)
        assert isinstance(exports, list), path
        assert all(isinstance(name, str) for name in exports), path
        return tuple(exports)
    raise AssertionError(f"{path} must assign a literal __all__")


def _load_modified_manifest(tmp_path: Path, *, old: str, new: str) -> CoverageManifest:
    source = (TESTS_ROOT / "contracts" / "coverage.toml").read_text(encoding="utf-8")
    modified = source.replace(old, new, 1)
    assert modified != source
    manifest_path = tmp_path / "coverage.toml"
    manifest_path.write_text(modified, encoding="utf-8")
    return load_coverage_manifest(manifest_path)


def _required_claims(manifest: CoverageManifest) -> list[CoverageClaim]:
    claims: list[CoverageClaim] = []
    for capability in manifest.capabilities:
        if capability.enforcement != "required":
            continue
        for evidence in capability.required_evidence:
            for collection_path in capability.collection_paths_for(evidence):
                source = next(PACKAGE_ROOT.glob(collection_path))
                claims.append(
                    CoverageClaim(
                        scenario=capability.scenario,
                        backend=capability.backend,
                        evidence=evidence,
                        source=(
                            source.relative_to(PACKAGE_ROOT).as_posix()
                            + "::synthetic_required_claim"
                        ),
                    )
                )
    return claims


class _EvidenceMarker:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs


class _EvidenceItem:
    def __init__(self, nodeid: str, *markers: _EvidenceMarker) -> None:
        self.nodeid = nodeid
        self._markers = markers

    def iter_markers(self, *, name: str) -> tuple[_EvidenceMarker, ...]:
        assert name == "evidence_for"
        return self._markers


class _EvidenceReport:
    def __init__(
        self,
        nodeid: str,
        when: str,
        outcome: str,
        claim: CoverageClaim,
        *,
        wasxfail: str | None = None,
    ) -> None:
        self.nodeid = nodeid
        self.when = when
        self.outcome = outcome
        self.user_properties = [
            (
                EVIDENCE_PROPERTY,
                (claim.scenario, claim.backend, claim.evidence, claim.source),
            )
        ]
        if wasxfail is not None:
            self.wasxfail = wasxfail


def _record_evidence_item(
    run: EvidenceRun,
    nodeid: str,
    claim: CoverageClaim,
    *,
    outcomes: tuple[str, str, str] = ("passed", "passed", "passed"),
    wasxfail: str | None = None,
    register_expected: bool = True,
) -> None:
    if register_expected:
        run.record_expected(nodeid, (claim,))
    for when, outcome in zip(("setup", "call", "teardown"), outcomes, strict=True):
        run.record_report(
            _EvidenceReport(
                nodeid,
                when,
                outcome,
                claim,
                wasxfail=wasxfail,
            )
        )


@pytest.mark.backend_core
def test_manifest_is_a_complete_initial_backend_map(
    coverage_manifest: CoverageManifest,
) -> None:
    assert coverage_manifest.mode == "migration"
    assert {capability.backend for capability in coverage_manifest.capabilities} == {
        "core",
        "cutlass",
        "numba_mlir",
    }
    assert {capability.status for capability in coverage_manifest.capabilities} == {
        "blocked",
        "executable",
    }


@pytest.mark.backend_core
@pytest.mark.evidence_for("contracts.manifest", backend="core", evidence="semantics")
def test_group_first_manifest_matches_common_profile_and_runtime_exports(
    coverage_manifest: CoverageManifest,
) -> None:
    common_operations = tuple(COMMON_PROFILE_MATRIX)
    expected_operations = {
        "core": common_operations,
        "cutlass": common_operations,
        "numba_mlir": common_operations,
    }
    runtime_exports = {
        "core": tuple(root_api.__all__),
        "cutlass": _literal_exports(
            PACKAGE_ROOT / "cuda" / "coop" / "cutlass" / "__init__.py"
        ),
        "numba_mlir": _literal_exports(
            PACKAGE_ROOT / "cuda" / "coop" / "numba_mlir" / "__init__.py"
        ),
    }

    for backend, operations in expected_operations.items():
        capabilities = {
            capability.scenario.removeprefix("group."): capability
            for capability in coverage_manifest.capabilities
            if capability.backend == backend and capability.value_model == "group_first"
        }

        assert tuple(capabilities) == operations
        assert set(operations) <= set(runtime_exports[backend])
        for operation in operations:
            capability = capabilities[operation]
            assert capability.public_surface == _GROUP_FIRST_SURFACES[backend]
            assert capability.status == "executable"
            assert capability.public_symbols == (operation,)
            if operation in COMMON_PROFILE_MATRIX:
                profile = COMMON_PROFILE_MATRIX[operation]
                assert (
                    capability.required_evidence
                    == profile["required_evidence"][backend]
                )
                assert (
                    capability.enforcement == profile["evidence_enforcement"][backend]
                )


def test_common_profile_matrix_synthesizes_each_role_once(
    coverage_manifest: CoverageManifest,
) -> None:
    common_capabilities = [
        capability
        for capability in coverage_manifest.capabilities
        if capability.backend in {"core", "cutlass", "numba_mlir"}
        and capability.value_model == "group_first"
        and capability.scenario.removeprefix("group.") in COMMON_PROFILE_MATRIX
    ]

    assert len(common_capabilities) == 3 * len(COMMON_PROFILE_MATRIX)
    assert {
        (capability.backend, capability.scenario) for capability in common_capabilities
    } == {
        (backend, f"group.{operation}")
        for backend in ("core", "cutlass", "numba_mlir")
        for operation in COMMON_PROFILE_MATRIX
    }


def test_common_profile_direct_ownership_uses_exact_operation_lanes(
    coverage_manifest: CoverageManifest,
) -> None:
    for operation in COMMON_PROFILE_MATRIX:
        profile = COMMON_PROFILE_MATRIX[operation]
        for role, lanes_by_evidence in profile["evidence_collection_paths"].items():
            capability = coverage_manifest.by_key[(f"group.{operation}", role)]

            assert capability.collection_paths == ()
            assert capability.enforcement == "required"
            for evidence, paths in lanes_by_evidence.items():
                assert capability.collection_paths_for(evidence) == paths
                assert all(
                    path.endswith(".py") and not any(token in path for token in "*?[")
                    for path in paths
                )

    sum_lanes = COMMON_PROFILE_MATRIX["sum"]["evidence_collection_paths"]
    for backend in ("cutlass", "numba_mlir"):
        assert sum_lanes[backend]["runtime"] == (
            f"tests/backends/{backend}/runtime/test_common_root_sum_scan.py",
            f"tests/backends/{backend}/runtime/test_common_profile.py",
        )


@pytest.mark.parametrize(
    ("policy_entry", "message"),
    [
        ('  "core",\n', "unknown backends.*core"),
        ('  "executable",\n', "requires the 'executable' status"),
        ('  "group_first",\n', "requires the 'group_first' value model"),
    ],
)
def test_common_profile_respects_manifest_policy(
    tmp_path: Path, policy_entry: str, message: str
) -> None:
    with pytest.raises(CoverageManifestError, match=message):
        _load_modified_manifest(tmp_path, old=policy_entry, new="")


def test_manifest_migration_sources_exist(
    coverage_manifest: CoverageManifest,
) -> None:
    assert migration_source_errors(coverage_manifest, package_root=PACKAGE_ROOT) == ()


def test_manifest_required_collection_paths_exist(
    coverage_manifest: CoverageManifest,
) -> None:
    assert (
        required_collection_path_errors(coverage_manifest, package_root=PACKAGE_ROOT)
        == ()
    )


def test_every_required_collection_path_must_resolve(
    coverage_manifest: CoverageManifest,
) -> None:
    required = next(
        capability
        for capability in coverage_manifest.capabilities
        if capability.enforcement == "required"
    )
    manifest = replace(
        coverage_manifest,
        capabilities=(
            replace(
                required,
                collection_paths_by_evidence=(
                    (
                        "semantics",
                        (
                            "tests/contracts/parity/test_coverage_manifest.py",
                            "tests/contracts/parity/test_removed_manifest_contract.py",
                        ),
                    ),
                ),
            ),
        ),
    )

    assert required_collection_path_errors(manifest, package_root=PACKAGE_ROOT) == (
        f"{required.scenario!r}/{required.backend} required 'semantics' "
        "collection path "
        "'tests/contracts/parity/test_removed_manifest_contract.py' does not resolve",
    )


def test_migration_collection_paths_remain_advisory(
    coverage_manifest: CoverageManifest,
) -> None:
    required = next(
        capability
        for capability in coverage_manifest.capabilities
        if capability.enforcement == "required"
    )
    manifest = replace(
        coverage_manifest,
        capabilities=(
            replace(
                required,
                enforcement="migration",
                collection_paths=("tests/backends/removed_backend.py",),
                collection_paths_by_evidence=(),
            ),
        ),
    )

    assert required_collection_path_errors(manifest, package_root=PACKAGE_ROOT) == ()


def test_unknown_claims_are_collection_errors(
    coverage_manifest: CoverageManifest,
) -> None:
    claims = _required_claims(coverage_manifest)
    claims.append(
        CoverageClaim(
            scenario="scoped.block.does_not_exist",
            backend="numba_mlir",
            evidence="runtime",
            source="unknown-test",
        )
    )

    audit = audit_coverage_claims(coverage_manifest, claims)

    assert any("unknown capability" in error for error in audit.errors)


def test_evidence_for_marker_creates_a_conformance_claim() -> None:
    item = _EvidenceItem(
        "tests/contracts/parity/test_fake.py::test_semantics[case]",
        _EvidenceMarker(
            "contracts.manifest",
            backend="core",
            evidence="semantics",
        ),
    )

    claims, errors = claims_from_items([item])

    assert errors == ()
    assert claims == (
        CoverageClaim(
            scenario="contracts.manifest",
            backend="core",
            evidence="semantics",
            source="tests/contracts/parity/test_fake.py::test_semantics",
        ),
    )


def test_invalid_evidence_for_marker_names_the_new_marker() -> None:
    item = _EvidenceItem(
        "tests/contracts/parity/test_fake.py::test_semantics",
        _EvidenceMarker(backend="core", evidence="semantics"),
    )

    claims, errors = claims_from_items([item])

    assert claims == ()
    assert errors == (
        "tests/contracts/parity/test_fake.py::test_semantics evidence_for marker "
        "requires one string scenario argument",
    )


def test_only_fully_passing_tests_earn_conformance_evidence() -> None:
    claim = CoverageClaim(
        scenario="contracts.manifest",
        backend="core",
        evidence="semantics",
        source="tests/contracts/parity/test_fake.py::test_semantics",
    )
    run = EvidenceRun()
    _record_evidence_item(
        run,
        "tests/contracts/parity/test_fake.py::test_semantics[passing]",
        claim,
    )

    assert run.earned_claims() == (claim,)


@pytest.mark.parametrize(
    ("outcomes", "wasxfail"),
    [
        (("skipped", "passed", "passed"), None),
        (("passed", "skipped", "passed"), None),
        (("passed", "passed", "skipped"), None),
        (("failed", "passed", "passed"), None),
        (("passed", "failed", "passed"), None),
        (("passed", "passed", "failed"), None),
        (("passed", "passed", "passed"), "unexpected pass"),
    ],
)
def test_skips_failures_and_xfail_do_not_earn_conformance_evidence(
    outcomes: tuple[str, str, str], wasxfail: str | None
) -> None:
    claim = CoverageClaim(
        scenario="contracts.manifest",
        backend="core",
        evidence="semantics",
        source="tests/contracts/parity/test_fake.py::test_semantics",
    )
    run = EvidenceRun()
    _record_evidence_item(
        run,
        "tests/contracts/parity/test_fake.py::test_semantics",
        claim,
        outcomes=outcomes,
        wasxfail=wasxfail,
    )

    assert run.earned_claims() == ()


def test_every_selected_parameter_case_must_pass_to_earn_one_logical_claim() -> None:
    claim = CoverageClaim(
        scenario="contracts.manifest",
        backend="core",
        evidence="semantics",
        source="tests/contracts/parity/test_fake.py::test_semantics",
    )
    run = EvidenceRun()
    _record_evidence_item(
        run,
        "tests/contracts/parity/test_fake.py::test_semantics[passing]",
        claim,
    )
    _record_evidence_item(
        run,
        "tests/contracts/parity/test_fake.py::test_semantics[skipped]",
        claim,
        outcomes=("skipped", "passed", "passed"),
    )

    assert run.earned_claims() == ()


def test_unreported_selected_parameter_case_does_not_earn_evidence() -> None:
    claim = CoverageClaim(
        scenario="contracts.manifest",
        backend="core",
        evidence="semantics",
        source="tests/contracts/parity/test_fake.py::test_semantics",
    )
    passing = "tests/contracts/parity/test_fake.py::test_semantics[passing]"
    unreported = "tests/contracts/parity/test_fake.py::test_semantics[unreported]"
    run = EvidenceRun()
    run.record_expected(passing, (claim,))
    run.record_expected(unreported, (claim,))
    _record_evidence_item(run, passing, claim)

    assert run.earned_claims() == ()


def test_xdist_expected_inventory_round_trips_to_the_controller() -> None:
    claim = CoverageClaim(
        scenario="contracts.manifest",
        backend="core",
        evidence="semantics",
        source="tests/contracts/parity/test_fake.py::test_semantics",
    )
    nodeid = "tests/contracts/parity/test_fake.py::test_semantics"
    worker = EvidenceRun()
    worker.record_expected(nodeid, (claim,))
    worker.record_selected_cells((claim.cell,))

    controller = EvidenceRun()
    _record_evidence_item(controller, nodeid, claim, register_expected=False)
    controller.merge_worker_inventory(
        "gw0",
        expected=worker.expected_payload(),
        selected_cells=worker.selected_cells,
        collected_nodeids=worker.collected_nodeids,
        selected_nodeids=worker.selected_nodeids,
        collection_errors=(),
    )

    assert controller.earned_claims() == (claim,)
    assert controller.selected_cells == (claim.cell,)
    assert controller.metadata_errors == ()


def test_report_only_claim_cannot_earn_evidence() -> None:
    claim = CoverageClaim(
        scenario="contracts.manifest",
        backend="core",
        evidence="semantics",
        source="tests/contracts/parity/test_fake.py::test_semantics",
    )
    nodeid = "tests/contracts/parity/test_fake.py::test_semantics"
    run = EvidenceRun()
    _record_evidence_item(run, nodeid, claim, register_expected=False)

    assert run.earned_claims() == ()
    assert run.metadata_errors == (
        f"{nodeid} reported cuda.coop evidence without a selected inventory entry",
    )


def test_inconsistent_xdist_worker_inventory_is_an_audit_error() -> None:
    first = EvidenceRun()
    second = EvidenceRun()
    claim = CoverageClaim(
        scenario="contracts.manifest",
        backend="core",
        evidence="semantics",
        source="tests/contracts/parity/test_fake.py::test_semantics",
    )
    first.record_expected("tests/test_fake.py::test_semantics[first]", (claim,))
    second.record_expected("tests/test_fake.py::test_semantics[second]", (claim,))
    controller = EvidenceRun()
    controller.merge_worker_inventory(
        "gw0",
        expected=first.expected_payload(),
        selected_cells=first.selected_cells,
        collected_nodeids=first.collected_nodeids,
        selected_nodeids=first.selected_nodeids,
        collection_errors=(),
    )
    controller.merge_worker_inventory(
        "gw1",
        expected=second.expected_payload(),
        selected_cells=second.selected_cells,
        collected_nodeids=second.collected_nodeids,
        selected_nodeids=second.selected_nodeids,
        collection_errors=(),
    )

    assert controller.metadata_errors == (
        "gw1: cuda.coop evidence inventory differs across xdist workers",
    )


def test_malformed_xdist_worker_inventory_is_an_audit_error() -> None:
    controller = EvidenceRun()

    controller.merge_worker_inventory(
        "gw0",
        expected=None,
        selected_cells=(),
        collected_nodeids=(),
        selected_nodeids=(),
        collection_errors=(),
    )

    assert controller.metadata_errors == (
        "gw0: malformed cuda.coop expected-evidence inventory",
    )


def test_xdist_worker_collection_errors_are_deferred_to_the_controller() -> None:
    controller = EvidenceRun()

    controller.merge_worker_inventory(
        "gw0",
        expected=(),
        selected_cells=(),
        collected_nodeids=(),
        selected_nodeids=(),
        collection_errors=("invalid evidence marker",),
    )

    assert controller.metadata_errors == ("invalid evidence marker",)


def test_multiple_tests_may_support_one_evidence_cell(
    coverage_manifest: CoverageManifest,
) -> None:
    claims = _required_claims(coverage_manifest)
    claims.append(
        CoverageClaim(
            scenario="contracts.manifest",
            backend="core",
            evidence="semantics",
            source=(
                "tests/contracts/parity/test_coverage_manifest.py"
                "::second_synthetic_test"
            ),
        )
    )

    audit = audit_coverage_claims(coverage_manifest, claims)

    assert audit.errors == ()


def test_required_multi_path_evidence_earns_each_selected_owner(
    coverage_manifest: CoverageManifest,
) -> None:
    capability = coverage_manifest.by_key[("group.sum", "cutlass")]
    manifest = replace(coverage_manifest, capabilities=(capability,))
    focused_path, extended_path = capability.collection_paths_for("runtime")
    focused_claim = CoverageClaim(
        scenario=capability.scenario,
        backend=capability.backend,
        evidence="runtime",
        source=f"{focused_path}::test_focused",
    )
    extended_claim = replace(
        focused_claim,
        source=f"{extended_path}::test_extended_groups",
    )

    partial = audit_coverage_claims(
        manifest,
        [focused_claim],
        selected_paths=(focused_path, extended_path),
    )
    focused_only = audit_coverage_claims(
        manifest,
        [focused_claim],
        selected_paths=(focused_path,),
    )
    complete = audit_coverage_claims(
        manifest,
        [focused_claim, extended_claim],
        selected_paths=(focused_path, extended_path),
    )

    assert partial.errors == (
        "missing 'runtime' conformance evidence for 'group.sum'/cutlass from "
        f"required collection path {extended_path!r}",
    )
    assert focused_only.errors == ()
    assert complete.errors == ()


def test_required_claim_must_come_from_its_declared_lane(
    coverage_manifest: CoverageManifest,
) -> None:
    claims = [
        CoverageClaim(
            scenario="contracts.manifest",
            backend="core",
            evidence="semantics",
            source="tests/backends/numba_mlir/unit/test_wrong_owner.py::test_claim",
        )
    ]

    audit = audit_coverage_claims(coverage_manifest, claims)

    assert any(
        "outside its declared collection lanes" in error for error in audit.errors
    )
    assert any(
        "missing 'semantics' conformance evidence for 'contracts.manifest'/core"
        == error
        for error in audit.errors
    )


def test_required_reduce_claim_must_come_from_its_declared_lane(
    coverage_manifest: CoverageManifest,
) -> None:
    claim = CoverageClaim(
        scenario="group.reduce",
        backend="numba_mlir",
        evidence="lowering",
        source="tests/backends/cutlass/unit/test_wrong_lane.py::test_claim",
    )

    audit = audit_coverage_claims(
        coverage_manifest,
        [claim],
        selected_paths=(),
        selected_cells=(claim.cell,),
    )

    assert any(
        "outside its declared collection lanes" in error for error in audit.errors
    )
    assert any(
        "missing 'lowering' conformance evidence for 'group.reduce'/numba_mlir" == error
        for error in audit.errors
    )


def test_required_common_profile_claim_must_use_its_exact_operation_lane(
    coverage_manifest: CoverageManifest,
) -> None:
    claim = CoverageClaim(
        scenario="group.load",
        backend="numba_mlir",
        evidence="lowering",
        source="tests/backends/numba_mlir/unit/test_group_api.py::test_claim",
    )

    audit = audit_coverage_claims(
        coverage_manifest,
        [claim],
        selected_paths=(),
        selected_cells=(claim.cell,),
    )

    assert any(
        "outside its declared collection lanes" in error for error in audit.errors
    )
    assert (
        "missing 'lowering' conformance evidence for 'group.load'/numba_mlir"
        in audit.errors
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_role", "must define operation-specific collection paths"),
        ("missing_evidence", "collection paths must define exactly"),
        ("wildcard", "must name an exact package-relative"),
        ("directory", "must name an exact package-relative"),
        ("absolute", "must name an exact package-relative"),
        ("parent", "must name an exact package-relative"),
        ("nodeid", "must name an exact package-relative"),
    ],
)
def test_required_common_profile_rejects_inexact_operation_lanes(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    matrix = dict(COMMON_PROFILE_MATRIX)
    profile = deepcopy(COMMON_PROFILE_MATRIX["load"])
    matrix["load"] = profile
    paths = profile["evidence_collection_paths"]
    numba_paths = paths["numba_mlir"]

    if mutation == "missing_role":
        del paths["numba_mlir"]
    elif mutation == "missing_evidence":
        del numba_paths["link"]
    else:
        numba_paths["link"] = {
            "wildcard": ("tests/providers/qualification/numba_mlir/**/*.py",),
            "directory": ("tests/providers/qualification/numba_mlir",),
            "absolute": ("/tests/providers/qualification/test_link.py",),
            "parent": ("tests/providers/../qualification/test_link.py",),
            "nodeid": ("tests/providers/qualification/test_link.py::test_link",),
        }[mutation]

    monkeypatch.setattr(coverage_support, "COMMON_PROFILE_MATRIX", matrix)
    with pytest.raises(CoverageManifestError, match=message):
        load_coverage_manifest(TESTS_ROOT / "contracts" / "coverage.toml")


def test_recursive_collection_owner_matches_zero_or_more_directories(
    coverage_manifest: CoverageManifest,
) -> None:
    required = next(
        capability
        for capability in coverage_manifest.capabilities
        if capability.enforcement == "required"
    )
    required = replace(
        required,
        collection_paths=(),
        collection_paths_by_evidence=(
            ("semantics", ("tests/contracts/**/test_*.py",)),
        ),
    )
    manifest = replace(coverage_manifest, capabilities=(required,))

    for source in (
        "tests/contracts/test_direct.py::test_claim",
        "tests/contracts/parity/test_nested.py::test_claim",
    ):
        claim = CoverageClaim(
            scenario=required.scenario,
            backend=required.backend,
            evidence="semantics",
            source=source,
        )
        assert audit_coverage_claims(manifest, [claim]).errors == ()


def test_directory_collection_owner_includes_descendant_tests(
    coverage_manifest: CoverageManifest,
) -> None:
    required = next(
        capability
        for capability in coverage_manifest.capabilities
        if capability.enforcement == "required"
    )
    required = replace(
        required,
        collection_paths=(),
        collection_paths_by_evidence=(("semantics", ("tests/contracts/parity",)),),
    )
    manifest = replace(coverage_manifest, capabilities=(required,))
    claim = CoverageClaim(
        scenario=required.scenario,
        backend=required.backend,
        evidence="semantics",
        source="tests/contracts/parity/test_nested.py::test_claim",
    )

    assert audit_coverage_claims(manifest, [claim]).errors == ()


def test_required_missing_claim_is_a_collection_error(
    coverage_manifest: CoverageManifest,
) -> None:
    audit = audit_coverage_claims(coverage_manifest, [])

    assert any(
        "missing 'semantics' conformance evidence for 'contracts.manifest'/core"
        == error
        for error in audit.errors
    )
    assert audit.migration_gaps


def test_required_missing_claim_is_enforced_for_its_full_collection_path(
    coverage_manifest: CoverageManifest,
) -> None:
    audit = audit_coverage_claims(
        coverage_manifest,
        [],
        selected_paths=("tests/contracts/parity/test_coverage_manifest.py",),
    )

    assert any(
        "missing 'semantics' conformance evidence for 'contracts.manifest'/core"
        == error
        for error in audit.errors
    )


def test_exact_selection_enforces_only_its_selected_evidence_cell(
    coverage_manifest: CoverageManifest,
) -> None:
    selected_cell = ("contracts.manifest", "core", "semantics")
    audit = audit_coverage_claims(
        coverage_manifest,
        [],
        selected_paths=(),
        selected_cells=(selected_cell,),
    )

    assert audit.errors == (
        "missing 'semantics' conformance evidence for 'contracts.manifest'/core",
    )


def test_required_evidence_is_enforced_only_in_its_owned_lane(
    coverage_manifest: CoverageManifest,
) -> None:
    required = next(
        capability
        for capability in coverage_manifest.capabilities
        if capability.enforcement == "required"
    )
    required = replace(
        required,
        required_evidence=("api", "runtime"),
        collection_paths=(),
        collection_paths_by_evidence=(
            ("api", ("tests/contracts/public_api/test_fake.py",)),
            ("runtime", ("tests/backends/numba_mlir/runtime/test_fake.py",)),
        ),
    )
    manifest = replace(coverage_manifest, capabilities=(required,))

    contracts_audit = audit_coverage_claims(
        manifest,
        [],
        selected_paths=("tests/contracts/public_api/test_fake.py",),
    )
    runtime_audit = audit_coverage_claims(
        manifest,
        [],
        selected_paths=("tests/backends/numba_mlir/runtime/test_fake.py",),
    )
    satisfied_contracts_audit = audit_coverage_claims(
        manifest,
        [
            CoverageClaim(
                scenario=required.scenario,
                backend=required.backend,
                evidence="api",
                source="tests/contracts/public_api/test_fake.py::test_api",
            )
        ],
        selected_paths=("tests/contracts/public_api/test_fake.py",),
    )

    assert contracts_audit.errors == (
        "missing 'api' conformance evidence for 'contracts.manifest'/core",
    )
    assert runtime_audit.errors == (
        "missing 'runtime' conformance evidence for 'contracts.manifest'/core",
    )
    assert satisfied_contracts_audit.errors == ()


def test_legacy_collection_paths_remain_the_default_for_required_evidence(
    tmp_path: Path,
) -> None:
    coverage_manifest = _load_modified_manifest(
        tmp_path,
        old=(
            "collection_paths_by_evidence = { semantics = "
            '["tests/contracts/parity/test_coverage_manifest.py"] }'
        ),
        new=('collection_paths = ["tests/contracts/parity/test_coverage_manifest.py"]'),
    )
    required = next(
        capability
        for capability in coverage_manifest.capabilities
        if capability.enforcement == "required"
    )

    assert required.collection_paths_by_evidence == ()
    assert required.collection_paths_for("semantics") == required.collection_paths


def test_capability_evidence_owner_overrides_its_surface_owner(
    tmp_path: Path,
) -> None:
    manifest = _load_modified_manifest(
        tmp_path,
        old=(
            '{ name = "manifest", symbols = ["coverage.toml"], '
            'required_evidence = ["semantics"] }'
        ),
        new=(
            '{ name = "manifest", symbols = ["coverage.toml"], '
            'required_evidence = ["semantics"], '
            "collection_paths_by_evidence = { semantics = "
            '["tests/contracts/core/test_core_bindings.py"] } }'
        ),
    )
    required = next(
        capability
        for capability in manifest.capabilities
        if capability.enforcement == "required"
    )

    assert required.collection_paths_for("semantics") == (
        "tests/contracts/core/test_core_bindings.py",
    )


def test_required_evidence_must_have_a_collection_lane(tmp_path: Path) -> None:
    with pytest.raises(
        CoverageManifestError,
        match="required evidence has no collection lane.*semantics",
    ):
        _load_modified_manifest(
            tmp_path,
            old=(
                "collection_paths_by_evidence = { semantics = "
                '["tests/contracts/parity/test_coverage_manifest.py"] }'
            ),
            new=(
                "collection_paths_by_evidence = { runtime = "
                '["tests/contracts/parity/test_coverage_manifest.py"] }'
            ),
        )


def test_targeted_sibling_node_does_not_activate_required_evidence() -> None:
    target = (
        "tests/contracts/parity/test_coverage_manifest.py"
        "::test_manifest_migration_sources_exist"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", target, "--collect-only", "-q"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr
    assert "1 test collected" in result.stdout


def test_targeted_owner_node_does_not_activate_sibling_evidence() -> None:
    target = (
        "tests/contracts/parity/test_coverage_manifest.py"
        "::test_manifest_is_a_complete_initial_backend_map"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", target, "--collect-only", "-q"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr
    assert "1 test collected" in result.stdout


def test_shared_paths_are_location_independent() -> None:
    assert (TESTS_ROOT / "contracts" / "coverage.toml").is_file()
    assert PACKAGE_ROOT == TESTS_ROOT.parent
    assert REPO_ROOT != Path.cwd() or (REPO_ROOT / "CMakePresets.json").is_file()
    assert (REPO_ROOT / "CMakePresets.json").is_file()


class _CollectedCase:
    def __init__(self, nodeid: str, *markers: str) -> None:
        self.nodeid = nodeid
        self._markers = frozenset(markers)

    def get_closest_marker(self, name: str):
        return object() if name in self._markers else None


def test_matrix_guard_rejects_unclassified_cartesian_products(
    coverage_manifest: CoverageManifest,
) -> None:
    cases = [
        _CollectedCase(f"tests/contracts/core/test_fake.py::test_matrix[{index}]")
        for index in range(33)
    ]

    errors = audit_matrix_sizes(coverage_manifest, cases)

    assert len(errors) == 1
    assert "expands to 33 pull-request cases" in errors[0]


@pytest.mark.parametrize("cadence", ["large", "qualification", "stress"])
def test_matrix_guard_accepts_explicit_non_pr_cadence(
    coverage_manifest: CoverageManifest, cadence: str
) -> None:
    cases = [
        _CollectedCase(
            f"tests/providers/qualification/test_fake.py::test_matrix[{index}]",
            cadence,
        )
        for index in range(33)
    ]

    assert audit_matrix_sizes(coverage_manifest, cases) == ()


def test_matrix_waivers_are_bounded_current_and_resolve(
    coverage_manifest: CoverageManifest,
) -> None:
    for waiver in coverage_manifest.matrix_waivers:
        assert 32 < waiver.max_cases < 64
        assert waiver.expires >= date.today()
        relative_path = waiver.source.split("::", 1)[0]
        assert (PACKAGE_ROOT / relative_path).is_file()
