# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Conformance-manifest loading and earned-evidence auditing."""

from __future__ import annotations

import fnmatch
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

from .cases.api_contracts import COMMON_PROFILE_MATRIX, COMMON_PROFILE_ROLE_METADATA

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


_IDENTIFIER = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)*$")
_DIAGNOSTIC_STATUSES = frozenset({"blocked", "placeholder", "unsupported"})
EVIDENCE_PROPERTY = "cuda_coop_evidence_for"


class CoverageManifestError(ValueError):
    """The coverage manifest is malformed or internally inconsistent."""


@dataclass(frozen=True)
class Capability:
    """One backend-specific semantic capability."""

    scenario: str
    backend: str
    public_surface: str
    public_symbols: tuple[str, ...]
    value_model: str
    status: str
    provider: str
    required_evidence: tuple[str, ...]
    enforcement: str
    collection_paths: tuple[str, ...]
    collection_paths_by_evidence: tuple[tuple[str, tuple[str, ...]], ...]
    diagnostic: str | None
    migration_sources: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.scenario, self.backend

    def collection_paths_for(self, evidence: str) -> tuple[str, ...]:
        """Return the declared collection lanes for one evidence cell."""

        for owned_evidence, paths in self.collection_paths_by_evidence:
            if owned_evidence == evidence:
                return paths
        return self.collection_paths


@dataclass(frozen=True)
class CoverageManifest:
    """Validated manifest policy and expanded capabilities."""

    schema_version: int
    mode: str
    allowed_backends: frozenset[str]
    allowed_evidence: frozenset[str]
    allowed_statuses: frozenset[str]
    allowed_value_models: frozenset[str]
    allowed_enforcement: frozenset[str]
    capabilities: tuple[Capability, ...]
    matrix_waivers: tuple[MatrixWaiver, ...]

    @property
    def by_key(self) -> Mapping[tuple[str, str], Capability]:
        return {capability.key: capability for capability in self.capabilities}


@dataclass(frozen=True)
class CoverageClaim:
    """One test-to-cell conformance-evidence association."""

    scenario: str
    backend: str
    evidence: str
    source: str

    @property
    def cell(self) -> tuple[str, str, str]:
        return self.scenario, self.backend, self.evidence


@dataclass(frozen=True)
class CoverageAudit:
    """Hard conformance errors and visible migration debt."""

    errors: tuple[str, ...]
    migration_gaps: tuple[str, ...]

    def format_errors(self) -> str:
        return "\n".join(f"- {error}" for error in self.errors)


@dataclass
class _EvidenceItemOutcome:
    claims: set[CoverageClaim]
    phases: dict[str, set[str]]
    was_xfail: bool = False

    @property
    def fully_passed(self) -> bool:
        return not self.was_xfail and self.phases == {
            "setup": {"passed"},
            "call": {"passed"},
            "teardown": {"passed"},
        }


class EvidenceRun:
    """Aggregate evidence-bearing pytest outcomes without importing pytest."""

    def __init__(self) -> None:
        self._items: dict[str, _EvidenceItemOutcome] = {}
        self._expected: dict[str, set[CoverageClaim]] = {}
        self._selected_cells: set[tuple[str, str, str]] = set()
        self._selected_nodeids: dict[str, None] = {}
        self._collected_nodeids: dict[str, None] = {}
        self._worker_inventory: tuple[Any, ...] | None = None
        self._metadata_errors: list[str] = []

    @property
    def nodeids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self._collected_nodeids, *self._items)))

    @property
    def collected_nodeids(self) -> tuple[str, ...]:
        return tuple(sorted(self._collected_nodeids))

    @property
    def selected_nodeids(self) -> tuple[str, ...]:
        return tuple(sorted(self._selected_nodeids))

    def record_collected_nodeid(self, nodeid: Any) -> None:
        """Record a collected path, including modules skipped at collection."""

        if isinstance(nodeid, str) and nodeid:
            self._collected_nodeids[nodeid] = None

    def record_selected_nodeid(self, nodeid: Any) -> None:
        """Record one item retained after pytest selection and deselection."""

        if isinstance(nodeid, str) and nodeid:
            self._selected_nodeids[nodeid] = None
            self.record_collected_nodeid(nodeid)

    def record_expected(self, nodeid: str, claims: Iterable[CoverageClaim]) -> None:
        """Record every selected case expected to support its logical claims."""

        claims = set(claims)
        if not claims:
            return
        self.record_selected_nodeid(nodeid)
        self._expected.setdefault(nodeid, set()).update(claims)

    def record_selected_cells(self, cells: Iterable[tuple[str, str, str]]) -> None:
        """Record required cells activated by exact-node selections."""

        self._selected_cells.update(cells)

    @property
    def selected_cells(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(sorted(self._selected_cells))

    def expected_payload(self) -> tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]:
        """Return an execnet-safe inventory for xdist controller aggregation."""

        return tuple(
            (
                nodeid,
                tuple(
                    sorted(
                        (
                            claim.scenario,
                            claim.backend,
                            claim.evidence,
                            claim.source,
                        )
                        for claim in claims
                    )
                ),
            )
            for nodeid, claims in sorted(self._expected.items())
        )

    @classmethod
    def _normalize_expected_payload(
        cls, payload: Any
    ) -> tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]:
        if not isinstance(payload, (list, tuple)):
            raise ValueError("malformed cuda.coop expected-evidence inventory")
        inventory: dict[str, set[CoverageClaim]] = {}
        for entry in payload:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError("malformed cuda.coop expected-evidence entry")
            nodeid, values = entry
            if not isinstance(nodeid, str) or not isinstance(values, (list, tuple)):
                raise ValueError("malformed cuda.coop expected-evidence entry")
            claims = {cls._claim_from_metadata(nodeid, value) for value in values}
            if not claims or nodeid in inventory:
                raise ValueError("malformed cuda.coop expected-evidence entry")
            inventory[nodeid] = claims
        return tuple(
            (
                nodeid,
                tuple(
                    sorted(
                        (
                            claim.scenario,
                            claim.backend,
                            claim.evidence,
                            claim.source,
                        )
                        for claim in claims
                    )
                ),
            )
            for nodeid, claims in sorted(inventory.items())
        )

    @staticmethod
    def _normalize_selected_cells_payload(
        payload: Any,
    ) -> tuple[tuple[str, str, str], ...]:
        if not isinstance(payload, (list, tuple)):
            raise ValueError("malformed cuda.coop selected-evidence cells")
        cells: set[tuple[str, str, str]] = set()
        for value in payload:
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 3
                or any(not isinstance(part, str) or not part for part in value)
            ):
                raise ValueError("malformed cuda.coop selected-evidence cell")
            scenario, backend, evidence = value
            cells.add((scenario, backend, evidence))
        return tuple(sorted(cells))

    @staticmethod
    def _normalize_collected_payload(payload: Any) -> tuple[str, ...]:
        if not isinstance(payload, (list, tuple)) or any(
            not isinstance(nodeid, str) for nodeid in payload
        ):
            raise ValueError("malformed cuda.coop collected-node inventory")
        return tuple(sorted(set(payload)))

    def merge_worker_inventory(
        self,
        worker: str,
        *,
        expected: Any,
        selected_cells: Any,
        collected_nodeids: Any,
        selected_nodeids: Any,
        collection_errors: Any,
    ) -> None:
        """Safely merge and cross-check one completed xdist worker inventory."""

        try:
            inventory = (
                self._normalize_expected_payload(expected),
                self._normalize_selected_cells_payload(selected_cells),
                self._normalize_collected_payload(collected_nodeids),
                self._normalize_collected_payload(selected_nodeids),
                self._normalize_errors_payload(collection_errors),
            )
        except ValueError as exc:
            self._metadata_errors.append(f"{worker}: {exc}")
            return
        if self._worker_inventory is not None:
            if inventory != self._worker_inventory:
                self._metadata_errors.extend(inventory[-1])
                self._metadata_errors.append(
                    f"{worker}: cuda.coop evidence inventory differs across "
                    "xdist workers"
                )
            return

        self._worker_inventory = inventory
        (
            expected_inventory,
            selected_inventory,
            collected_inventory,
            selected_node_inventory,
            worker_errors,
        ) = inventory
        for nodeid, values in expected_inventory:
            claims = tuple(self._claim_from_metadata(nodeid, value) for value in values)
            self.record_expected(nodeid, claims)
        self.record_selected_cells(selected_inventory)
        for nodeid in collected_inventory:
            self.record_collected_nodeid(nodeid)
        for nodeid in selected_node_inventory:
            self.record_selected_nodeid(nodeid)
        self._metadata_errors.extend(worker_errors)

    @staticmethod
    def _normalize_errors_payload(payload: Any) -> tuple[str, ...]:
        if not isinstance(payload, (list, tuple)) or any(
            not isinstance(error, str) or not error for error in payload
        ):
            raise ValueError("malformed cuda.coop collection-error inventory")
        return tuple(dict.fromkeys(payload))

    def record_report(self, report: Any) -> None:
        """Record one setup, call, or teardown report."""

        nodeid = getattr(report, "nodeid", None)
        when = getattr(report, "when", None)
        outcome = getattr(report, "outcome", None)
        if not isinstance(nodeid, str) or when not in {"setup", "call", "teardown"}:
            return
        if outcome not in {"passed", "failed", "skipped"}:
            return

        self.record_collected_nodeid(nodeid)
        values = tuple(
            value
            for name, value in getattr(report, "user_properties", ())
            if name == EVIDENCE_PROPERTY
        )
        if not values:
            return
        try:
            claims = {self._claim_from_metadata(nodeid, value) for value in values}
        except ValueError as exc:
            self._metadata_errors.append(str(exc))
            return

        item = self._items.setdefault(nodeid, _EvidenceItemOutcome(set(), {}))
        item.claims.update(claims)
        item.phases.setdefault(when, set()).add(outcome)
        item.was_xfail |= getattr(report, "wasxfail", None) is not None

    @staticmethod
    def _claim_from_metadata(nodeid: str, value: Any) -> CoverageClaim:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 4
            or any(not isinstance(part, str) or not part for part in value)
        ):
            raise ValueError(f"{nodeid} has malformed {EVIDENCE_PROPERTY!r} metadata")
        scenario, backend, evidence, source = value
        return CoverageClaim(scenario, backend, evidence, source)

    def earned_claims(self) -> tuple[CoverageClaim, ...]:
        """Return claims whose selected logical test passed in every phase."""

        items_by_claim: dict[CoverageClaim, list[_EvidenceItemOutcome | None]] = {}
        for nodeid, claims in self._expected.items():
            item = self._items.get(nodeid)
            if item is not None and item.claims != claims:
                item = None
            for claim in claims:
                items_by_claim.setdefault(claim, []).append(item)
        return tuple(
            sorted(
                (
                    claim
                    for claim, items in items_by_claim.items()
                    if items
                    and all(item is not None and item.fully_passed for item in items)
                ),
                key=lambda claim: (
                    claim.scenario,
                    claim.backend,
                    claim.evidence,
                    claim.source,
                ),
            )
        )

    @property
    def metadata_errors(self) -> tuple[str, ...]:
        """Return malformed, unexpected, or inconsistent evidence metadata."""

        errors = list(self._metadata_errors)
        for nodeid, item in sorted(self._items.items()):
            expected = self._expected.get(nodeid)
            if expected is None:
                errors.append(
                    f"{nodeid} reported cuda.coop evidence without a selected "
                    "inventory entry"
                )
            elif item.claims != expected:
                errors.append(
                    f"{nodeid} reported cuda.coop evidence that differs from its "
                    "selected inventory entry"
                )
        return tuple(dict.fromkeys(errors))


@dataclass(frozen=True)
class MatrixWaiver:
    """A bounded, expiring waiver for one legacy Cartesian test function."""

    source: str
    max_cases: int
    reason: str
    expires: date


def _as_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise CoverageManifestError(f"{context} must be a nonempty string")
    return value


def _as_string_tuple(value: Any, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise CoverageManifestError(f"{context} must be a list of nonempty strings")
    if len(value) != len(set(value)):
        raise CoverageManifestError(f"{context} contains duplicate entries")
    return tuple(value)


def _policy_set(policy: Mapping[str, Any], name: str) -> frozenset[str]:
    values = _as_string_tuple(policy.get(name), context=f"policy.{name}")
    if not values:
        raise CoverageManifestError(f"policy.{name} must not be empty")
    return frozenset(values)


def _as_collection_paths_by_evidence(
    value: Any,
    *,
    context: str,
    allowed_evidence: frozenset[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(value, dict):
        raise CoverageManifestError(f"{context} must be a table")

    result: list[tuple[str, tuple[str, ...]]] = []
    for evidence, raw_paths in value.items():
        evidence = _as_string(evidence, context=f"{context} key")
        if evidence not in allowed_evidence:
            raise CoverageManifestError(f"{context} has unknown evidence {evidence!r}")
        paths = _as_string_tuple(
            raw_paths,
            context=f"{context}.{evidence}",
        )
        if not paths:
            raise CoverageManifestError(f"{context}.{evidence} must not be empty")
        result.append((evidence, paths))
    return tuple(sorted(result))


def _validate_exact_test_collection_paths(
    paths: tuple[str, ...],
    *,
    context: str,
) -> None:
    for path in paths:
        pure_path = PurePosixPath(path)
        if (
            any(character in path for character in "*?[")
            or "\\" in path
            or "::" in path
            or pure_path.is_absolute()
            or pure_path.as_posix() != path
            or not pure_path.parts
            or pure_path.parts[0] != "tests"
            or ".." in pure_path.parts
            or pure_path.suffix != ".py"
            or not pure_path.name.startswith("test_")
        ):
            raise CoverageManifestError(
                f"{context} required evidence collection path {path!r} must "
                "name an exact package-relative tests/**/test_*.py file"
            )


def _format_diagnostic(template: str | None, *, scenario: str, operation: str):
    if template is None:
        return None
    try:
        return template.format(scenario=scenario, operation=operation)
    except KeyError as exc:
        raise CoverageManifestError(
            f"{scenario} diagnostic has unknown template field {exc.args[0]!r}"
        ) from exc


def _expand_surface(
    raw_surface: Mapping[str, Any],
    *,
    surface_index: int,
    policy: Mapping[str, Any],
    allowed_backends: frozenset[str],
    allowed_evidence: frozenset[str],
    allowed_statuses: frozenset[str],
    allowed_value_models: frozenset[str],
    allowed_enforcement: frozenset[str],
) -> list[Capability]:
    context = f"surface[{surface_index}]"
    backend = _as_string(raw_surface.get("backend"), context=f"{context}.backend")
    prefix = _as_string(raw_surface.get("scope"), context=f"{context}.scope")
    public_surface = _as_string(
        raw_surface.get("public_surface"), context=f"{context}.public_surface"
    )
    value_model = _as_string(
        raw_surface.get("value_model"), context=f"{context}.value_model"
    )
    provider = _as_string(raw_surface.get("provider"), context=f"{context}.provider")
    enforcement = _as_string(
        raw_surface.get("enforcement"), context=f"{context}.enforcement"
    )
    migration_sources = _as_string_tuple(
        raw_surface.get("migration_sources", []),
        context=f"{context}.migration_sources",
    )
    collection_paths = _as_string_tuple(
        raw_surface.get("collection_paths", []),
        context=f"{context}.collection_paths",
    )
    collection_paths_by_evidence = _as_collection_paths_by_evidence(
        raw_surface.get("collection_paths_by_evidence", {}),
        context=f"{context}.collection_paths_by_evidence",
        allowed_evidence=allowed_evidence,
    )
    diagnostic_template = raw_surface.get("diagnostic")
    if diagnostic_template is not None:
        diagnostic_template = _as_string(
            diagnostic_template, context=f"{context}.diagnostic"
        )

    if backend not in allowed_backends:
        raise CoverageManifestError(f"{context} has unknown backend {backend!r}")
    if value_model not in allowed_value_models:
        raise CoverageManifestError(
            f"{context} has unknown value model {value_model!r}"
        )
    if enforcement not in allowed_enforcement:
        raise CoverageManifestError(
            f"{context} has unknown enforcement {enforcement!r}"
        )
    if not _IDENTIFIER.fullmatch(prefix):
        raise CoverageManifestError(f"{context}.scope is not a stable identifier")
    if enforcement == "migration" and not migration_sources:
        raise CoverageManifestError(
            f"{context} migration surfaces must name legacy evidence sources"
        )
    required_by_status = policy.get("required_evidence")
    if not isinstance(required_by_status, dict):
        raise CoverageManifestError("policy.required_evidence must be a table")

    capabilities: list[Capability] = []
    for status in allowed_statuses:
        raw_capabilities = raw_surface.get(status, [])
        if not isinstance(raw_capabilities, list):
            raise CoverageManifestError(f"{context}.{status} must be a list")

        for capability_index, raw_capability in enumerate(raw_capabilities):
            capability_context = f"{context}.{status}[{capability_index}]"
            if isinstance(raw_capability, str):
                operation = raw_capability
                public_symbols = (operation,)
                operation_provider = provider
                operation_enforcement = enforcement
                operation_sources = migration_sources
                operation_collection_paths = collection_paths
                operation_collection_paths_by_evidence = collection_paths_by_evidence
                diagnostic = diagnostic_template
                required_evidence = required_by_status.get(status)
            elif isinstance(raw_capability, dict):
                operation = _as_string(
                    raw_capability.get("name"), context=f"{capability_context}.name"
                )
                public_symbols = _as_string_tuple(
                    raw_capability.get("symbols", [operation]),
                    context=f"{capability_context}.symbols",
                )
                operation_provider = _as_string(
                    raw_capability.get("provider", provider),
                    context=f"{capability_context}.provider",
                )
                operation_enforcement = _as_string(
                    raw_capability.get("enforcement", enforcement),
                    context=f"{capability_context}.enforcement",
                )
                operation_sources = _as_string_tuple(
                    raw_capability.get("migration_sources", list(migration_sources)),
                    context=f"{capability_context}.migration_sources",
                )
                operation_collection_paths = _as_string_tuple(
                    raw_capability.get("collection_paths", list(collection_paths)),
                    context=f"{capability_context}.collection_paths",
                )
                operation_collection_path_overrides = _as_collection_paths_by_evidence(
                    raw_capability.get("collection_paths_by_evidence", {}),
                    context=(f"{capability_context}.collection_paths_by_evidence"),
                    allowed_evidence=allowed_evidence,
                )
                merged_collection_paths_by_evidence = dict(collection_paths_by_evidence)
                merged_collection_paths_by_evidence.update(
                    operation_collection_path_overrides
                )
                operation_collection_paths_by_evidence = tuple(
                    sorted(merged_collection_paths_by_evidence.items())
                )
                diagnostic = raw_capability.get("diagnostic", diagnostic_template)
                if diagnostic is not None:
                    diagnostic = _as_string(
                        diagnostic, context=f"{capability_context}.diagnostic"
                    )
                required_evidence = raw_capability.get(
                    "required_evidence", required_by_status.get(status)
                )
            else:
                raise CoverageManifestError(
                    f"{capability_context} must be a string or table"
                )

            scenario = f"{prefix}.{operation}"
            if not _IDENTIFIER.fullmatch(scenario):
                raise CoverageManifestError(
                    f"{capability_context} produces invalid scenario {scenario!r}"
                )
            if operation_enforcement not in allowed_enforcement:
                raise CoverageManifestError(
                    f"{capability_context} has unknown enforcement "
                    f"{operation_enforcement!r}"
                )
            if operation_enforcement == "migration" and not operation_sources:
                raise CoverageManifestError(
                    f"{capability_context} migration capability has no source"
                )
            evidence = _as_string_tuple(
                required_evidence,
                context=f"{capability_context}.required_evidence",
            )
            unknown_evidence = set(evidence) - allowed_evidence
            if unknown_evidence:
                raise CoverageManifestError(
                    f"{capability_context} has unknown evidence "
                    f"{sorted(unknown_evidence)!r}"
                )
            evidence_lanes = dict(operation_collection_paths_by_evidence)
            missing_collection_lanes = [
                required
                for required in evidence
                if not evidence_lanes.get(required, operation_collection_paths)
            ]
            if operation_enforcement == "required" and missing_collection_lanes:
                raise CoverageManifestError(
                    f"{capability_context} required evidence has no collection "
                    f"lane: {missing_collection_lanes!r}"
                )

            formatted_diagnostic = _format_diagnostic(
                diagnostic, scenario=scenario, operation=operation
            )
            if status in _DIAGNOSTIC_STATUSES and formatted_diagnostic is None:
                raise CoverageManifestError(
                    f"{capability_context} must specify its exact diagnostic"
                )

            capabilities.append(
                Capability(
                    scenario=scenario,
                    backend=backend,
                    public_surface=public_surface,
                    public_symbols=public_symbols,
                    value_model=value_model,
                    status=status,
                    provider=operation_provider,
                    required_evidence=evidence,
                    enforcement=operation_enforcement,
                    collection_paths=operation_collection_paths,
                    collection_paths_by_evidence=(
                        operation_collection_paths_by_evidence
                    ),
                    diagnostic=formatted_diagnostic,
                    migration_sources=operation_sources,
                )
            )

    if not capabilities:
        raise CoverageManifestError(f"{context} contains no capabilities")
    return capabilities


def expand_common_profile_capabilities(
    *,
    allowed_backends: frozenset[str],
    allowed_evidence: frozenset[str],
    allowed_enforcement: frozenset[str],
    allowed_statuses: frozenset[str],
    allowed_value_models: frozenset[str],
) -> tuple[Capability, ...]:
    """Expand the authoritative common V1 matrix into evidence obligations."""

    if "executable" not in allowed_statuses:
        raise CoverageManifestError(
            "common profile requires the 'executable' status to be allowed"
        )
    if "group_first" not in allowed_value_models:
        raise CoverageManifestError(
            "common profile requires the 'group_first' value model to be allowed"
        )

    capabilities: list[Capability] = []
    for operation, profile in COMMON_PROFILE_MATRIX.items():
        if not _IDENTIFIER.fullmatch(operation):
            raise CoverageManifestError(
                f"common profile operation {operation!r} is not a stable identifier"
            )

        certified_backends = profile.get("certified_backends")
        if not isinstance(certified_backends, tuple) or any(
            not isinstance(backend, str) or not backend
            for backend in certified_backends
        ):
            raise CoverageManifestError(
                f"common profile {operation!r} has invalid certified_backends"
            )
        roles = ("core", *certified_backends)
        unknown_backends = set(roles) - allowed_backends
        if unknown_backends:
            raise CoverageManifestError(
                f"common profile {operation!r} has unknown backends "
                f"{sorted(unknown_backends)!r}"
            )

        required_by_role = profile.get("required_evidence")
        operation_lanes_by_role = profile.get("evidence_collection_paths", {})
        enforcement_by_role = profile.get("evidence_enforcement")
        if not isinstance(required_by_role, dict) or tuple(required_by_role) != roles:
            raise CoverageManifestError(
                f"common profile {operation!r} must define evidence for {roles!r}"
            )
        if (
            not isinstance(enforcement_by_role, dict)
            or tuple(enforcement_by_role) != roles
        ):
            raise CoverageManifestError(
                f"common profile {operation!r} must define enforcement for {roles!r}"
            )
        if not isinstance(operation_lanes_by_role, dict) or (
            set(operation_lanes_by_role) - set(roles)
        ):
            raise CoverageManifestError(
                f"common profile {operation!r} has invalid evidence collection "
                "path roles"
            )

        for role in roles:
            metadata = COMMON_PROFILE_ROLE_METADATA.get(role)
            if not isinstance(metadata, dict):
                raise CoverageManifestError(
                    f"common profile role {role!r} has no metadata"
                )

            required_evidence = required_by_role[role]
            if not isinstance(required_evidence, tuple) or any(
                not isinstance(evidence, str) or not evidence
                for evidence in required_evidence
            ):
                raise CoverageManifestError(
                    f"common profile {operation!r}/{role} has invalid evidence"
                )
            unknown_evidence = set(required_evidence) - allowed_evidence
            if unknown_evidence:
                raise CoverageManifestError(
                    f"common profile {operation!r}/{role} has unknown evidence "
                    f"{sorted(unknown_evidence)!r}"
                )

            enforcement = enforcement_by_role[role]
            if enforcement not in allowed_enforcement:
                raise CoverageManifestError(
                    f"common profile {operation!r}/{role} has unknown enforcement "
                    f"{enforcement!r}"
                )

            default_lanes = metadata.get("collection_paths_by_evidence")
            if not isinstance(default_lanes, dict):
                raise CoverageManifestError(
                    f"common profile role {role!r} has invalid collection lanes"
                )
            operation_lanes = operation_lanes_by_role.get(role, {})
            if not isinstance(operation_lanes, dict) or (
                set(operation_lanes) - set(required_evidence)
            ):
                raise CoverageManifestError(
                    f"common profile {operation!r}/{role} has invalid evidence "
                    "collection paths"
                )
            if enforcement == "required":
                if role not in operation_lanes_by_role:
                    raise CoverageManifestError(
                        f"common profile {operation!r}/{role} required evidence "
                        "must define operation-specific collection paths"
                    )
                if set(operation_lanes) != set(required_evidence):
                    raise CoverageManifestError(
                        f"common profile {operation!r}/{role} required evidence "
                        "collection paths must define exactly "
                        f"{required_evidence!r}"
                    )
            lanes: list[tuple[str, tuple[str, ...]]] = []
            for evidence in required_evidence:
                raw_paths = operation_lanes.get(
                    evidence,
                    default_lanes.get(evidence),
                )
                if (
                    not isinstance(raw_paths, tuple)
                    or not raw_paths
                    or any(not isinstance(path, str) or not path for path in raw_paths)
                ):
                    raise CoverageManifestError(
                        f"common profile role {role!r} has no {evidence!r} "
                        "collection lane"
                    )
                if enforcement == "required":
                    _validate_exact_test_collection_paths(
                        raw_paths,
                        context=f"common profile {operation!r}/{role}/{evidence}",
                    )
                lanes.append((evidence, raw_paths))

            migration_sources = metadata.get("migration_sources")
            if not isinstance(migration_sources, tuple) or not migration_sources:
                raise CoverageManifestError(
                    f"common profile role {role!r} has no migration sources"
                )
            public_surface = metadata.get("public_surface")
            provider = metadata.get("provider")
            if not isinstance(public_surface, str) or not public_surface:
                raise CoverageManifestError(
                    f"common profile role {role!r} has no public surface"
                )
            if not isinstance(provider, str) or not provider:
                raise CoverageManifestError(
                    f"common profile role {role!r} has no provider"
                )

            capabilities.append(
                Capability(
                    scenario=f"group.{operation}",
                    backend=role,
                    public_surface=public_surface,
                    public_symbols=(operation,),
                    value_model="group_first",
                    status="executable",
                    provider=provider,
                    required_evidence=required_evidence,
                    enforcement=enforcement,
                    collection_paths=(),
                    collection_paths_by_evidence=tuple(lanes),
                    diagnostic=None,
                    migration_sources=migration_sources,
                )
            )

    return tuple(capabilities)


def load_coverage_manifest(path: Path) -> CoverageManifest:
    """Load and validate ``coverage.toml`` without importing a DSL backend."""

    with path.open("rb") as stream:
        raw = tomllib.load(stream)

    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise CoverageManifestError(
            f"unsupported coverage manifest schema version {schema_version!r}"
        )

    policy = raw.get("policy")
    if not isinstance(policy, dict):
        raise CoverageManifestError("coverage manifest requires a policy table")

    mode = _as_string(policy.get("mode"), context="policy.mode")
    allowed_backends = _policy_set(policy, "allowed_backends")
    allowed_evidence = _policy_set(policy, "allowed_evidence")
    allowed_statuses = _policy_set(policy, "allowed_statuses")
    allowed_value_models = _policy_set(policy, "allowed_value_models")
    allowed_enforcement = _policy_set(policy, "allowed_enforcement")

    raw_surfaces = raw.get("surface")
    if not isinstance(raw_surfaces, list) or not raw_surfaces:
        raise CoverageManifestError("coverage manifest requires at least one surface")

    common_profile_capabilities = expand_common_profile_capabilities(
        allowed_backends=allowed_backends,
        allowed_evidence=allowed_evidence,
        allowed_enforcement=allowed_enforcement,
        allowed_statuses=allowed_statuses,
        allowed_value_models=allowed_value_models,
    )
    capabilities: list[Capability] = []
    common_profile_inserted = False
    for surface_index, raw_surface in enumerate(raw_surfaces):
        if not isinstance(raw_surface, dict):
            raise CoverageManifestError(f"surface[{surface_index}] must be a table")
        if not common_profile_inserted and raw_surface.get("scope") == "group":
            capabilities.extend(common_profile_capabilities)
            common_profile_inserted = True
        capabilities.extend(
            _expand_surface(
                raw_surface,
                surface_index=surface_index,
                policy=policy,
                allowed_backends=allowed_backends,
                allowed_evidence=allowed_evidence,
                allowed_statuses=allowed_statuses,
                allowed_value_models=allowed_value_models,
                allowed_enforcement=allowed_enforcement,
            )
        )
    if not common_profile_inserted:
        capabilities.extend(common_profile_capabilities)

    duplicate_keys = [
        key
        for key, count in Counter(capability.key for capability in capabilities).items()
        if count > 1
    ]
    if duplicate_keys:
        raise CoverageManifestError(
            f"duplicate capability rows: {sorted(duplicate_keys)!r}"
        )

    raw_waivers = raw.get("matrix_waiver", [])
    if not isinstance(raw_waivers, list):
        raise CoverageManifestError("matrix_waiver must be an array of tables")

    matrix_waivers: list[MatrixWaiver] = []
    for waiver_index, raw_waiver in enumerate(raw_waivers):
        context = f"matrix_waiver[{waiver_index}]"
        if not isinstance(raw_waiver, dict):
            raise CoverageManifestError(f"{context} must be a table")
        source = _as_string(raw_waiver.get("source"), context=f"{context}.source")
        reason = _as_string(raw_waiver.get("reason"), context=f"{context}.reason")
        max_cases = raw_waiver.get("max_cases")
        if (
            not isinstance(max_cases, int)
            or isinstance(max_cases, bool)
            or max_cases <= 32
        ):
            raise CoverageManifestError(
                f"{context}.max_cases must be an integer greater than 32"
            )
        raw_expires = _as_string(
            raw_waiver.get("expires"), context=f"{context}.expires"
        )
        try:
            expires = date.fromisoformat(raw_expires)
        except ValueError as exc:
            raise CoverageManifestError(
                f"{context}.expires must use YYYY-MM-DD"
            ) from exc
        matrix_waivers.append(
            MatrixWaiver(
                source=source,
                max_cases=max_cases,
                reason=reason,
                expires=expires,
            )
        )
    duplicate_waivers = [
        source
        for source, count in Counter(waiver.source for waiver in matrix_waivers).items()
        if count > 1
    ]
    if duplicate_waivers:
        raise CoverageManifestError(
            f"duplicate matrix waivers: {sorted(duplicate_waivers)!r}"
        )

    return CoverageManifest(
        schema_version=schema_version,
        mode=mode,
        allowed_backends=allowed_backends,
        allowed_evidence=allowed_evidence,
        allowed_statuses=allowed_statuses,
        allowed_value_models=allowed_value_models,
        allowed_enforcement=allowed_enforcement,
        capabilities=tuple(capabilities),
        matrix_waivers=tuple(matrix_waivers),
    )


def audit_coverage_claims(
    manifest: CoverageManifest,
    claims: Iterable[CoverageClaim],
    *,
    selected_paths: Iterable[str] | None = None,
    selected_cells: Iterable[tuple[str, str, str]] = (),
) -> CoverageAudit:
    """Audit conformance-evidence claims, collection lanes, and required cells."""

    claims = tuple(claims)
    capabilities = manifest.by_key
    errors: list[str] = []

    valid_claims: list[CoverageClaim] = []
    for claim in claims:
        capability = capabilities.get((claim.scenario, claim.backend))
        if capability is None:
            errors.append(
                f"{claim.source} claims unknown capability "
                f"{claim.scenario!r} for backend {claim.backend!r}"
            )
            continue
        if claim.evidence not in manifest.allowed_evidence:
            errors.append(
                f"{claim.source} claims unknown evidence {claim.evidence!r} for "
                f"{claim.scenario!r}/{claim.backend}"
            )
            continue
        collection_lanes = capability.collection_paths_for(claim.evidence)
        if (
            claim.evidence in capability.required_evidence
            and collection_lanes
            and not _matches_collection_path(claim.source, collection_lanes)
        ):
            errors.append(
                f"{claim.source} claims {claim.evidence!r} conformance evidence for "
                f"{claim.scenario!r}/{claim.backend} outside its declared "
                "collection lanes"
            )
            continue
        valid_claims.append(claim)

    claimed_cells = {claim.cell for claim in valid_claims}
    claimed_sources_by_cell: dict[tuple[str, str, str], set[str]] = {}
    for claim in valid_claims:
        claimed_sources_by_cell.setdefault(claim.cell, set()).add(claim.source)
    selected_paths = None if selected_paths is None else tuple(selected_paths)
    selected_cells = frozenset(selected_cells)
    migration_gaps: list[str] = []
    for capability in manifest.capabilities:
        for evidence in capability.required_evidence:
            cell = capability.scenario, capability.backend, evidence
            collection_paths = capability.collection_paths_for(evidence)
            message = (
                f"missing {evidence!r} conformance evidence for "
                f"{capability.scenario!r}/{capability.backend}"
            )
            enforce_required = capability.enforcement == "required" and (
                selected_paths is None
                or cell in selected_cells
                or any(
                    _matches_collection_path(
                        path,
                        capability.collection_paths_for(evidence),
                    )
                    for path in selected_paths
                )
            )
            if cell not in claimed_cells and enforce_required:
                errors.append(message)
                continue
            if cell not in claimed_cells and capability.enforcement == "migration":
                migration_gaps.append(message)
                continue
            if not enforce_required:
                continue

            # Exact paths on a required cell are cumulative owners, not
            # alternatives. When a test invocation selects more than one of
            # those paths, every selected owner must earn the claim. This lets
            # one operation split backend evidence across distinct support
            # routes without allowing the first passing route to mask a
            # deselected or skipped sibling route.
            activated_paths = (
                collection_paths
                if selected_paths is None
                else tuple(
                    collection_path
                    for collection_path in collection_paths
                    if any(
                        _matches_collection_path(path, (collection_path,))
                        for path in selected_paths
                    )
                )
            )
            claimed_sources = claimed_sources_by_cell[cell]
            for collection_path in activated_paths:
                if any(
                    _matches_collection_path(source, (collection_path,))
                    for source in claimed_sources
                ):
                    continue
                errors.append(
                    f"{message} from required collection path {collection_path!r}"
                )

    return CoverageAudit(tuple(errors), tuple(migration_gaps))


def migration_source_errors(
    manifest: CoverageManifest, *, package_root: Path
) -> tuple[str, ...]:
    """Return migration rows whose declared legacy evidence cannot be found."""

    errors: list[str] = []
    for capability in manifest.capabilities:
        if capability.enforcement != "migration":
            continue
        if not any(
            next(package_root.glob(pattern), None) is not None
            for pattern in capability.migration_sources
        ):
            errors.append(
                f"{capability.scenario!r}/{capability.backend} has no matching "
                f"migration source in {list(capability.migration_sources)!r}"
            )
    return tuple(errors)


def required_collection_path_errors(
    manifest: CoverageManifest, *, package_root: Path
) -> tuple[str, ...]:
    """Return required coverage paths that no longer resolve in the test tree."""

    errors: list[str] = []
    for capability in manifest.capabilities:
        if capability.enforcement != "required":
            continue
        for evidence in capability.required_evidence:
            for pattern in capability.collection_paths_for(evidence):
                if next(package_root.glob(pattern), None) is None:
                    errors.append(
                        f"{capability.scenario!r}/{capability.backend} required "
                        f"{evidence!r} collection path {pattern!r} does not resolve"
                    )
    return tuple(errors)


def audit_matrix_sizes(
    manifest: CoverageManifest,
    items: Sequence[Any],
    *,
    today: date | None = None,
) -> tuple[str, ...]:
    """Reject accidental pull-request matrices above 32 collected cases."""

    managed_roots = (
        "tests/backends/",
        "tests/contracts/",
        "tests/integration/",
        "tests/providers/",
    )
    exempt_markers = ("large", "qualification", "stress")
    cases_by_source: Counter[str] = Counter()
    for item in items:
        source = _logical_test_source(item.nodeid)
        if not source.startswith(managed_roots):
            continue
        if any(
            item.get_closest_marker(marker) is not None for marker in exempt_markers
        ):
            continue
        cases_by_source[source] += 1

    waiver_by_source = {waiver.source: waiver for waiver in manifest.matrix_waivers}
    today = date.today() if today is None else today
    errors: list[str] = []
    for source, count in sorted(cases_by_source.items()):
        if count <= 32:
            continue
        waiver = waiver_by_source.get(source)
        if waiver is None:
            errors.append(
                f"{source} expands to {count} pull-request cases; limit it to 32, "
                "mark it stress/large/qualification, or add a bounded migration "
                "waiver to coverage.toml"
            )
            continue
        if today > waiver.expires:
            errors.append(
                f"{source} matrix waiver expired on {waiver.expires.isoformat()}"
            )
        if count > waiver.max_cases:
            errors.append(
                f"{source} expands to {count} cases, above its migration waiver "
                f"limit of {waiver.max_cases}"
            )
    return tuple(errors)


def claims_from_items(
    items: Sequence[Any],
) -> tuple[tuple[CoverageClaim, ...], tuple[str, ...]]:
    """Parse ``evidence_for`` markers without importing pytest here."""

    claims: set[CoverageClaim] = set()
    errors: list[str] = []
    for item in items:
        source = _logical_test_source(item.nodeid)
        for marker in item.iter_markers(name="evidence_for"):
            if len(marker.args) != 1 or not isinstance(marker.args[0], str):
                errors.append(
                    f"{source} evidence_for marker requires one string scenario "
                    "argument"
                )
                continue
            unknown_keywords = set(marker.kwargs) - {"backend", "evidence"}
            if unknown_keywords:
                errors.append(
                    f"{source} evidence_for marker has unknown keywords "
                    f"{sorted(unknown_keywords)!r}"
                )
                continue
            backend = marker.kwargs.get("backend")
            evidence = marker.kwargs.get("evidence")
            if not isinstance(backend, str) or not backend:
                errors.append(
                    f"{source} evidence_for marker requires a nonempty backend string"
                )
                continue
            if not isinstance(evidence, str) or not evidence:
                errors.append(
                    f"{source} evidence_for marker requires a nonempty evidence string"
                )
                continue
            claims.add(
                CoverageClaim(
                    scenario=marker.args[0],
                    backend=backend,
                    evidence=evidence,
                    source=source,
                )
            )
    return tuple(claims), tuple(errors)


def _logical_test_source(nodeid: str) -> str:
    """Return a package-relative logical test function ID."""

    source = nodeid.split("[", 1)[0]
    tests_index = source.find("tests/")
    return source[tests_index:] if tests_index >= 0 else source


def _matches_collection_path(source: str, patterns: Iterable[str]) -> bool:
    """Return whether a test node belongs to one declared collection lane."""

    source_path = PurePosixPath(source.split("::", 1)[0])
    source_parts = source_path.parts

    def match_parts(
        remaining_source: tuple[str, ...],
        remaining_pattern: tuple[str, ...],
    ) -> bool:
        if not remaining_pattern:
            return not remaining_source
        if remaining_pattern[0] == "**":
            return match_parts(remaining_source, remaining_pattern[1:]) or (
                bool(remaining_source)
                and match_parts(remaining_source[1:], remaining_pattern)
            )
        return (
            bool(remaining_source)
            and fnmatch.fnmatchcase(
                remaining_source[0],
                remaining_pattern[0],
            )
            and match_parts(remaining_source[1:], remaining_pattern[1:])
        )

    for pattern in patterns:
        pattern_path = PurePosixPath(pattern)
        if not any(character in pattern for character in "*?["):
            if source_path == pattern_path or pattern_path in source_path.parents:
                return True
            continue
        if match_parts(source_parts, pattern_path.parts):
            return True
    return False


__all__ = [
    "Capability",
    "CoverageAudit",
    "CoverageClaim",
    "CoverageManifest",
    "CoverageManifestError",
    "EVIDENCE_PROPERTY",
    "EvidenceRun",
    "MatrixWaiver",
    "audit_matrix_sizes",
    "audit_coverage_claims",
    "claims_from_items",
    "expand_common_profile_capabilities",
    "load_coverage_manifest",
    "migration_source_errors",
    "required_collection_path_errors",
]
