# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import-light CUTLASS cooperative audit and roadmap metadata.

The registry describes the builtin ``cuda.coop.cutlass`` block and warp
surfaces.  It deliberately separates the C++ API selected for a future or
current group-first lowering from the implementation used by the scoped
compatibility API today.  A nominal CUB model in :mod:`cuda.coop._core` does
not by itself make a scoped provider CUB-backed.

Scoped-provider operand forms and currently modeled group-first operand forms
are recorded independently.  ``planned_target`` names the official C++ target
for the roadmap even when ``group_first_readiness`` says the shared planner or
provider conversion is still missing.

``planned_target`` and ``planned_api`` describe the default group-first route.
Selector-dependent alternate routes are recorded separately.  For example,
full Reduce uses CUDAX while partial-count and explicit-algorithm variants use
direct CUB.  Full physical-warp scalar Scan uses the canonical group-first CUB
provider, while logical-warp, partial, and ``ThreadData`` scoped forms remain
explicit compatibility routes.

The private test seam ``_api.register_provider_impl`` can replace a scoped
implementation at runtime, so the provenance below always means the builtin
provider shipped by CCCL.

This private module is a source-tree snapshot for tests, documentation, and
implementation planning.  It is not a runtime availability gate or a stable
compatibility API.  In particular, planner/provider code remains authoritative
for dtype, operator, launch-shape, valid-item, template, architecture, and
toolkit constraints.  ``group_first_planner_models_binding`` reports only that
the shared planner models an operation/selector combination; callers must also
inspect readiness and complete the ordered remaining stages before treating an
operation as exposed or executable.

Update this inventory whenever a scoped export, builtin provider, shared plan,
or C++ target changes.  ``READY`` describes implementation readiness in this
private source inventory; it is not a package-publication or release gate.  A
transition to ``READY`` requires planner tests, provider materialization, root
exposure, and primitive-specific compile, runtime, and final-link validation.
``BLOCKED_PROVIDER_PARITY`` records a materialized provider and its provenance,
but leaves those focused correctness checks as an explicit remaining stage.
Direct-provider SASS and resource comparisons are optional diagnostics rather
than readiness gates.
Unknown enum values raise ``ValueError`` and unknown binding names raise
``KeyError`` through the normal lookup APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from cuda.coop._core.group_dispatch import GroupLoweringTarget

from ._limits import MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD


class OperationFamily(str, Enum):
    ADJACENT_DIFFERENCE = "adjacent_difference"
    DISCONTINUITY = "discontinuity"
    EXCHANGE = "exchange"
    HISTOGRAM = "histogram"
    LOAD = "load"
    MERGE_SORT = "merge_sort"
    RADIX_RANK = "radix_rank"
    RADIX_SORT = "radix_sort"
    REDUCE = "reduce"
    ROW_REDUCE = "row_reduce"
    RUN_LENGTH_DECODE = "run_length_decode"
    SCAN = "scan"
    SHUFFLE = "shuffle"
    STORE = "store"
    TOPK = "topk"


class GroupKind(str, Enum):
    THREAD = "thread"
    BLOCK = "block"
    CLUSTER = "cluster"
    GRID = "grid"
    THREADS_WITHIN_WARP = "threads_within_warp"
    WARP = "warp"
    WARPS_WITHIN_BLOCK = "warps_within_block"


class OperandKind(str, Enum):
    SCALAR = "scalar"
    THREAD_DATA = "thread_data"
    TENSOR = "tensor"


class ApiStability(str, Enum):
    PUBLIC = "public"
    EXPERIMENTAL = "experimental"
    DETAIL = "detail"
    UNVERIFIED = "unverified"


class ApiAvailability(str, Enum):
    IN_TREE = "in_tree"
    NOT_IN_TREE = "not_in_tree"


class ProvenanceKind(str, Enum):
    CUDAX_PUBLIC = "cudax_public"
    CUB_PUBLIC = "cub_public"
    CUB_PUBLIC_WITH_GENERATED_ADAPTER = "cub_public_with_generated_adapter"
    CUB_DETAIL = "cub_detail"
    CUB_NOT_IN_TREE = "cub_not_in_tree"
    GENERATED_HANDWRITTEN = "generated_handwritten"
    CUTE_INDEXING = "cute_indexing"
    PAYLOAD_ADAPTER = "payload_adapter"


class GroupFirstReadiness(str, Enum):
    READY = "ready"
    READY_NOT_EXPOSED = "ready_not_exposed"
    BLOCKED_PROVIDER_PARITY = "blocked_provider_parity"
    BLOCKED_PROVIDER_CONVERSION = "blocked_provider_conversion"
    BLOCKED_PLANNER = "blocked_planner"
    BLOCKED_PLANNER_AND_DEPENDENCY = "blocked_planner_and_dependency"
    BLOCKED_PLANNER_AND_PROVIDER = "blocked_planner_and_provider"
    UNSUPPORTED = "unsupported"


class GroupFirstStage(str, Enum):
    DEPENDENCY = "dependency"
    PLANNER = "planner"
    PROVIDER = "provider"
    ROOT_EXPOSURE = "root_exposure"
    PARITY_VALIDATION = "parity_validation"


class ExportBindingKind(str, Enum):
    OPERATION = "operation"
    ALIAS = "alias"
    FACTORY = "factory"
    STATEFUL_ADAPTER = "stateful_adapter"
    SUPPORT = "support"


@dataclass(frozen=True)
class OperandForm:
    kind: OperandKind
    min_items_per_thread: int | None = None
    max_items_per_thread: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind is not OperandKind.THREAD_DATA and (
            self.min_items_per_thread is not None
            or self.max_items_per_thread is not None
        ):
            raise ValueError("item-count bounds apply only to ThreadData operands")
        minimum = self.min_items_per_thread
        maximum = self.max_items_per_thread
        if minimum is not None and minimum < 1:
            raise ValueError("min_items_per_thread must be positive")
        if maximum is not None and maximum < 1:
            raise ValueError("max_items_per_thread must be positive")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum item count exceeds maximum item count")


@dataclass(frozen=True)
class SelectorSupport:
    name: str
    accepted_values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("selector support name must be nonempty")
        object.__setattr__(self, "accepted_values", tuple(self.accepted_values))
        if not self.accepted_values or any(not value for value in self.accepted_values):
            raise ValueError("selector support requires accepted values")
        if len(set(self.accepted_values)) != len(self.accepted_values):
            raise ValueError("selector support values must be unique")


@dataclass(frozen=True)
class AlternateLoweringRouteCase:
    """One machine-checkable selector case for an alternate C++ route."""

    name: str
    operand_kinds: tuple[OperandKind, ...]
    selector_values: tuple[tuple[str, tuple[str, ...]], ...]
    min_items_per_thread: int | None = None
    max_items_per_thread: int | None = None
    requires_commutative_operator: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("alternate lowering route cases require a name")
        object.__setattr__(self, "operand_kinds", tuple(self.operand_kinds))
        if not self.operand_kinds:
            raise ValueError("alternate lowering route cases require operand kinds")
        if any(not isinstance(kind, OperandKind) for kind in self.operand_kinds):
            raise TypeError("route case operand_kinds must contain OperandKind values")
        if len(set(self.operand_kinds)) != len(self.operand_kinds):
            raise ValueError("route case operand kinds must be unique")
        if (
            self.min_items_per_thread is not None
            or self.max_items_per_thread is not None
        ) and OperandKind.THREAD_DATA not in self.operand_kinds:
            raise ValueError(
                "route case item-count bounds require a ThreadData operand"
            )
        minimum = self.min_items_per_thread
        maximum = self.max_items_per_thread
        if minimum is not None and minimum < 1:
            raise ValueError("route case minimum item count must be positive")
        if maximum is not None and maximum < 1:
            raise ValueError("route case maximum item count must be positive")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("route case minimum item count exceeds maximum")
        if not isinstance(self.requires_commutative_operator, bool):
            raise TypeError("requires_commutative_operator must be a bool")

        normalized_selectors = tuple(
            (selector_name, tuple(values))
            for selector_name, values in self.selector_values
        )
        object.__setattr__(self, "selector_values", normalized_selectors)
        selector_names = [name for name, _ in normalized_selectors]
        if not selector_names or any(not name for name in selector_names):
            raise ValueError("route cases require named selectors")
        if len(set(selector_names)) != len(selector_names):
            raise ValueError("route case selector names must be unique")
        if any(
            not values or any(not value for value in values)
            for _, values in normalized_selectors
        ):
            raise ValueError("route case selectors require accepted values")

    @property
    def operand_forms(self) -> tuple[OperandForm, ...]:
        """Return route-local operand forms, including conditional item bounds."""

        return tuple(
            OperandForm(
                kind,
                min_items_per_thread=(
                    self.min_items_per_thread
                    if kind is OperandKind.THREAD_DATA
                    else None
                ),
                max_items_per_thread=(
                    self.max_items_per_thread
                    if kind is OperandKind.THREAD_DATA
                    else None
                ),
            )
            for kind in self.operand_kinds
        )

    def matches(
        self,
        operand_kind: OperandKind | str,
        selectors: Mapping[str, str],
        *,
        items_per_thread: int | None = None,
        operator_commutative: bool | None = None,
    ) -> bool:
        """Return whether an operand and normalized selectors select this case."""

        try:
            operand_kind = OperandKind(operand_kind)
        except ValueError:
            return False
        expected_names = {name for name, _ in self.selector_values}
        if operand_kind not in self.operand_kinds or set(selectors) != expected_names:
            return False
        has_item_bounds = (
            self.min_items_per_thread is not None
            or self.max_items_per_thread is not None
        )
        if (
            operand_kind is OperandKind.THREAD_DATA
            and has_item_bounds
            and items_per_thread is None
        ):
            return False
        if operand_kind is OperandKind.THREAD_DATA and items_per_thread is not None:
            if isinstance(items_per_thread, bool) or not isinstance(
                items_per_thread, int
            ):
                return False
            if (
                self.min_items_per_thread is not None
                and items_per_thread < self.min_items_per_thread
            ):
                return False
            if (
                self.max_items_per_thread is not None
                and items_per_thread > self.max_items_per_thread
            ):
                return False
        if self.requires_commutative_operator and operator_commutative is not True:
            return False
        return all(
            selectors[name] in accepted for name, accepted in self.selector_values
        )


@dataclass(frozen=True)
class AlternateLoweringRoute:
    name: str
    cases: tuple[AlternateLoweringRouteCase, ...]
    target: GroupLoweringTarget
    api: CppApi
    provenance: ProviderProvenance | None = None
    readiness: GroupFirstReadiness = GroupFirstReadiness.READY
    unsupported_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("alternate lowering routes require a name")
        object.__setattr__(self, "cases", tuple(self.cases))
        if not self.cases or any(
            not isinstance(case, AlternateLoweringRouteCase) for case in self.cases
        ):
            raise ValueError("alternate lowering routes require typed route cases")
        case_names = [case.name for case in self.cases]
        if len(set(case_names)) != len(case_names):
            raise ValueError("alternate lowering route case names must be unique")
        if self.target is GroupLoweringTarget.UNSUPPORTED:
            raise ValueError("alternate lowering routes require a library target")
        if self.provenance is not None:
            if not isinstance(self.provenance, ProviderProvenance):
                raise TypeError("route provenance must be a ProviderProvenance")
            if self.provenance.api != self.api:
                raise ValueError("route provenance must describe the route API")
        object.__setattr__(self, "readiness", GroupFirstReadiness(self.readiness))
        if self.readiness is GroupFirstReadiness.READY:
            if self.unsupported_reason is not None:
                raise ValueError("ready alternate routes cannot have a reason")
        elif not self.unsupported_reason:
            raise ValueError("non-ready alternate routes require a reason")

    def matches(
        self,
        operand_kind: OperandKind | str,
        selectors: Mapping[str, str],
        *,
        items_per_thread: int | None = None,
        operator_commutative: bool | None = None,
    ) -> bool:
        """Return whether any typed case selects this alternate route."""

        return any(
            case.matches(
                operand_kind,
                selectors,
                items_per_thread=items_per_thread,
                operator_commutative=operator_commutative,
            )
            for case in self.cases
        )


@dataclass(frozen=True)
class CppApi:
    headers: tuple[str, ...]
    entity: str
    methods: tuple[str, ...]
    stability: ApiStability
    availability: ApiAvailability = ApiAvailability.IN_TREE

    def __post_init__(self) -> None:
        if not self.headers or any(not header for header in self.headers):
            raise ValueError("C++ API headers must be nonempty")
        if len(set(self.headers)) != len(self.headers):
            raise ValueError("C++ API headers must be unique")
        if not self.entity:
            raise ValueError("C++ API entity must be nonempty")
        if not self.methods or any(not method for method in self.methods):
            raise ValueError("C++ API methods must be nonempty")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("C++ API methods must be unique")
        object.__setattr__(self, "availability", ApiAvailability(self.availability))


@dataclass(frozen=True)
class ProviderProvenance:
    kind: ProvenanceKind
    note: str
    api: CppApi | None = None

    def __post_init__(self) -> None:
        if not self.note:
            raise ValueError("provider provenance requires a note")
        if (
            self.kind
            in {
                ProvenanceKind.CUDAX_PUBLIC,
                ProvenanceKind.CUB_PUBLIC,
                ProvenanceKind.CUB_PUBLIC_WITH_GENERATED_ADAPTER,
                ProvenanceKind.CUB_DETAIL,
                ProvenanceKind.CUB_NOT_IN_TREE,
            }
            and self.api is None
        ):
            raise ValueError("library-backed provenance requires a C++ API")
        if (
            self.kind
            in {
                ProvenanceKind.GENERATED_HANDWRITTEN,
                ProvenanceKind.CUTE_INDEXING,
                ProvenanceKind.PAYLOAD_ADAPTER,
            }
            and self.api is not None
        ):
            raise ValueError("non-library provenance must not claim a C++ API")


@dataclass(frozen=True)
class PayloadAdapterProvenance:
    """One non-C++ scoped compatibility route retained around a provider."""

    name: str
    provenance: ProviderProvenance

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("payload adapter provenance requires a name")
        if self.provenance.kind is not ProvenanceKind.PAYLOAD_ADAPTER:
            raise ValueError(
                "payload adapter provenance must use ProvenanceKind.PAYLOAD_ADAPTER"
            )


def _validate_route_metadata(
    selector_support: tuple[SelectorSupport, ...],
    alternate_routes: tuple[AlternateLoweringRoute, ...],
    *,
    selector_field: str,
    route_field: str,
) -> tuple[tuple[SelectorSupport, ...], tuple[AlternateLoweringRoute, ...]]:
    selectors = tuple(selector_support)
    if any(not isinstance(selector, SelectorSupport) for selector in selectors):
        raise TypeError(f"{selector_field} must contain SelectorSupport records")
    selector_names = [selector.name for selector in selectors]
    if len(selector_names) != len(set(selector_names)):
        raise ValueError(f"{selector_field} names must be unique")

    routes = tuple(alternate_routes)
    if any(not isinstance(route, AlternateLoweringRoute) for route in routes):
        raise TypeError(f"{route_field} must contain AlternateLoweringRoute records")
    route_names = [route.name for route in routes]
    if len(route_names) != len(set(route_names)):
        raise ValueError(f"{route_field} names must be unique")

    selector_values = {
        selector.name: {*selector.accepted_values, "omitted"} for selector in selectors
    }
    expected_selector_names = set(selector_values)
    for route in routes:
        for case in route.cases:
            case_selectors = dict(case.selector_values)
            if set(case_selectors) != expected_selector_names:
                raise ValueError(
                    f"{route_field} case selectors must exactly match {selector_field}"
                )
            if any(
                not set(values) <= selector_values[name]
                for name, values in case_selectors.items()
            ):
                raise ValueError(
                    f"{route_field} case values must be declared by "
                    f"{selector_field} or use 'omitted'"
                )
    return selectors, routes


@dataclass(frozen=True)
class Capability:
    """One family/group snapshot, not a runtime availability decision."""

    family: OperationFamily
    group: GroupKind
    builtin_scoped_operand_forms: tuple[OperandForm, ...]
    group_first_operand_forms: tuple[OperandForm, ...]
    planned_target: GroupLoweringTarget
    planned_api: CppApi | None
    builtin_scoped_provenance: ProviderProvenance | None
    primary_group_first_readiness: GroupFirstReadiness
    unsupported_reason: str | None
    group_first_provenance: ProviderProvenance | None = None
    group_first_selector_support: tuple[SelectorSupport, ...] = ()
    alternate_group_first_routes: tuple[AlternateLoweringRoute, ...] = ()
    builtin_scoped_selector_support: tuple[SelectorSupport, ...] = ()
    alternate_builtin_scoped_routes: tuple[AlternateLoweringRoute, ...] = ()
    scoped_payload_adapter_provenance: tuple[PayloadAdapterProvenance, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "group", GroupKind(self.group))
        has_scoped_surface = self.group in {GroupKind.BLOCK, GroupKind.WARP}
        for name, forms, allow_empty in (
            (
                "builtin_scoped_operand_forms",
                self.builtin_scoped_operand_forms,
                not has_scoped_surface,
            ),
            ("group_first_operand_forms", self.group_first_operand_forms, True),
        ):
            object.__setattr__(self, name, tuple(forms))
            if not forms and not allow_empty:
                raise ValueError("capabilities require scoped operand forms")
            if any(not isinstance(form, OperandForm) for form in forms):
                raise TypeError(f"{name} must contain OperandForm records")
            if len({form.kind for form in forms}) != len(forms):
                raise ValueError(f"operand kinds must be unique within {name}")
        if has_scoped_surface:
            if not isinstance(self.builtin_scoped_provenance, ProviderProvenance):
                raise TypeError("block and warp capabilities require scoped provenance")
        elif self.builtin_scoped_provenance is not None:
            raise ValueError(
                "root-only group capabilities cannot claim scoped provenance"
            )

        object.__setattr__(
            self,
            "primary_group_first_readiness",
            GroupFirstReadiness(self.primary_group_first_readiness),
        )
        is_ready = self.primary_group_first_readiness is GroupFirstReadiness.READY
        has_group_first_provider = is_ready or (
            self.primary_group_first_readiness
            is GroupFirstReadiness.BLOCKED_PROVIDER_PARITY
        )
        if is_ready:
            if self.unsupported_reason is not None:
                raise ValueError(
                    "ready capabilities must not have an unsupported reason"
                )
        elif not self.unsupported_reason:
            raise ValueError("non-ready capabilities require an unsupported reason")

        if has_group_first_provider:
            if self.group_first_provenance is None:
                raise ValueError(
                    "materialized group-first providers require group-first provenance"
                )
        else:
            if self.group_first_provenance is not None:
                raise ValueError(
                    "capabilities without a provider must not claim group-first provenance"
                )

        target_is_unsupported = self.planned_target is GroupLoweringTarget.UNSUPPORTED
        readiness_is_unsupported = (
            self.primary_group_first_readiness is GroupFirstReadiness.UNSUPPORTED
        )
        if target_is_unsupported != readiness_is_unsupported:
            raise ValueError("unsupported targets and unsupported readiness must agree")
        if target_is_unsupported:
            if self.planned_api is not None:
                raise ValueError(
                    "unsupported capabilities must not claim a planned API"
                )
        elif self.planned_api is None:
            raise ValueError("planned library targets require a C++ API")

        planner_is_ready = self.primary_group_first_readiness in {
            GroupFirstReadiness.READY,
            GroupFirstReadiness.READY_NOT_EXPOSED,
            GroupFirstReadiness.BLOCKED_PROVIDER_PARITY,
            GroupFirstReadiness.BLOCKED_PROVIDER_CONVERSION,
        }
        if planner_is_ready != bool(self.group_first_operand_forms):
            raise ValueError(
                "group-first operand forms must agree with planner readiness"
            )
        group_selectors, group_routes = _validate_route_metadata(
            self.group_first_selector_support,
            self.alternate_group_first_routes,
            selector_field="group_first_selector_support",
            route_field="alternate_group_first_routes",
        )
        object.__setattr__(self, "group_first_selector_support", group_selectors)
        object.__setattr__(self, "alternate_group_first_routes", group_routes)
        if self.group_first_selector_support and not planner_is_ready:
            raise ValueError("selector support requires a modeled group-first planner")
        if self.alternate_group_first_routes and not planner_is_ready:
            raise ValueError("alternate routes require a modeled group-first planner")

        scoped_selectors, scoped_routes = _validate_route_metadata(
            self.builtin_scoped_selector_support,
            self.alternate_builtin_scoped_routes,
            selector_field="builtin_scoped_selector_support",
            route_field="alternate_builtin_scoped_routes",
        )
        object.__setattr__(self, "builtin_scoped_selector_support", scoped_selectors)
        object.__setattr__(self, "alternate_builtin_scoped_routes", scoped_routes)
        if not has_scoped_surface and (scoped_selectors or scoped_routes):
            raise ValueError(
                "root-only group capabilities cannot declare scoped routes"
            )
        payload_adapters = tuple(self.scoped_payload_adapter_provenance)
        if any(
            not isinstance(adapter, PayloadAdapterProvenance)
            for adapter in payload_adapters
        ):
            raise TypeError(
                "scoped_payload_adapter_provenance must contain "
                "PayloadAdapterProvenance records"
            )
        adapter_names = [adapter.name for adapter in payload_adapters]
        if len(adapter_names) != len(set(adapter_names)):
            raise ValueError("scoped payload adapter names must be unique")
        object.__setattr__(
            self,
            "scoped_payload_adapter_provenance",
            payload_adapters,
        )
        if not has_scoped_surface and payload_adapters:
            raise ValueError(
                "root-only group capabilities cannot declare scoped adapters"
            )

    @property
    def group_first_readiness(self) -> GroupFirstReadiness:
        """Return conservative readiness across the primary and alternate routes."""

        if self.primary_group_first_readiness is not GroupFirstReadiness.READY:
            return self.primary_group_first_readiness
        for route in self.alternate_group_first_routes:
            if route.readiness is not GroupFirstReadiness.READY:
                return route.readiness
        return GroupFirstReadiness.READY

    @property
    def remaining_group_first_stages(self) -> tuple[GroupFirstStage, ...]:
        """Ordered gates remaining before this family can become ``READY``."""

        return {
            GroupFirstReadiness.READY: (),
            GroupFirstReadiness.READY_NOT_EXPOSED: (
                GroupFirstStage.ROOT_EXPOSURE,
                GroupFirstStage.PARITY_VALIDATION,
            ),
            GroupFirstReadiness.BLOCKED_PROVIDER_PARITY: (
                GroupFirstStage.PARITY_VALIDATION,
            ),
            GroupFirstReadiness.BLOCKED_PROVIDER_CONVERSION: (
                GroupFirstStage.PROVIDER,
                GroupFirstStage.ROOT_EXPOSURE,
                GroupFirstStage.PARITY_VALIDATION,
            ),
            GroupFirstReadiness.BLOCKED_PLANNER: (
                GroupFirstStage.PLANNER,
                GroupFirstStage.ROOT_EXPOSURE,
                GroupFirstStage.PARITY_VALIDATION,
            ),
            GroupFirstReadiness.BLOCKED_PLANNER_AND_DEPENDENCY: (
                GroupFirstStage.DEPENDENCY,
                GroupFirstStage.PLANNER,
                GroupFirstStage.ROOT_EXPOSURE,
                GroupFirstStage.PARITY_VALIDATION,
            ),
            GroupFirstReadiness.BLOCKED_PLANNER_AND_PROVIDER: (
                GroupFirstStage.PLANNER,
                GroupFirstStage.PROVIDER,
                GroupFirstStage.ROOT_EXPOSURE,
                GroupFirstStage.PARITY_VALIDATION,
            ),
            GroupFirstReadiness.UNSUPPORTED: (),
        }[self.group_first_readiness]


@dataclass(frozen=True)
class GroupMethodCapability:
    """Readiness of group observability and synchronization for one group kind."""

    group: GroupKind
    query_levels: tuple[str, ...]
    membership: bool
    synchronization: bool
    readiness: GroupFirstReadiness
    unsupported_reason: str | None = None
    validation_evidence: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "group", GroupKind(self.group))
        levels = tuple(self.query_levels)
        allowed_levels = {"thread", "warp", "block", "cluster", "grid"}
        if not levels or any(level not in allowed_levels for level in levels):
            raise ValueError("group method query levels must be hierarchy levels")
        if len(set(levels)) != len(levels):
            raise ValueError("group method query levels must be unique")
        object.__setattr__(self, "query_levels", levels)
        if not isinstance(self.membership, bool) or not isinstance(
            self.synchronization, bool
        ):
            raise TypeError("group method support flags must be bools")
        object.__setattr__(self, "readiness", GroupFirstReadiness(self.readiness))
        if self.readiness is GroupFirstReadiness.READY:
            if self.unsupported_reason is not None:
                raise ValueError("ready group methods cannot have a reason")
        elif not self.unsupported_reason:
            raise ValueError("non-ready group methods require a reason")
        if self.validation_evidence is not None and not isinstance(
            self.validation_evidence, str
        ):
            raise TypeError("validation_evidence must be a string or None")


@dataclass(frozen=True)
class ExportBinding:
    name: str
    kind: ExportBindingKind
    family: OperationFamily | None = None
    target_export: str | None = None
    selectors: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("export binding names must be nonempty")
        if len({name for name, _ in self.selectors}) != len(self.selectors):
            raise ValueError("export binding selector names must be unique")
        if self.kind is ExportBindingKind.SUPPORT:
            if self.family is not None or self.target_export is not None:
                raise ValueError("support exports do not bind an operation")
            return
        if self.family is None:
            raise ValueError("callable export bindings require an operation family")
        if (
            self.kind
            in {
                ExportBindingKind.ALIAS,
                ExportBindingKind.FACTORY,
                ExportBindingKind.STATEFUL_ADAPTER,
            }
            and not self.target_export
        ):
            raise ValueError("adapters require a target export")
        if self.kind is ExportBindingKind.OPERATION and self.target_export is not None:
            raise ValueError("canonical operations must not have a target export")


_SCALAR = OperandForm(OperandKind.SCALAR)
_THREAD_DATA = OperandForm(OperandKind.THREAD_DATA, min_items_per_thread=1)
_SHUFFLE_SCALAR = OperandForm(
    OperandKind.SCALAR,
    note="public-CUB Offset and Rotate modes only",
)
_SHUFFLE_THREAD_DATA = OperandForm(
    OperandKind.THREAD_DATA,
    min_items_per_thread=1,
    note="public-CUB Up and Down unit-shift modes only",
)
_THREAD_DATA_EXCHANGE = OperandForm(
    OperandKind.THREAD_DATA,
    min_items_per_thread=1,
    max_items_per_thread=MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD,
    note=(
        "canonical and block-scoped Exchange support x1 through x5; "
        "logical/scatter WarpExchange compatibility routes retain x4"
    ),
)
_TENSOR = OperandForm(OperandKind.TENSOR)
_GROUP_EXCHANGE_MODES = (
    SelectorSupport(
        "mode",
        ("striped_to_blocked", "blocked_to_striped"),
    ),
)

_GROUP_ADJACENT_DIFFERENCE_SELECTORS = (
    SelectorSupport("direction", ("left", "right")),
    SelectorSupport("valid_items", ("omitted", "runtime")),
    SelectorSupport("tile_predecessor_item", ("omitted", "runtime")),
    SelectorSupport("tile_successor_item", ("omitted", "runtime")),
)
_GROUP_DISCONTINUITY_SELECTORS = (
    SelectorSupport("mode", ("heads", "tails", "heads_and_tails")),
    SelectorSupport("tile_predecessor_item", ("omitted", "runtime")),
    SelectorSupport("tile_successor_item", ("omitted", "runtime")),
)
_GROUP_SHUFFLE_SELECTORS = (
    SelectorSupport("mode", ("offset", "rotate", "up", "down")),
    SelectorSupport("distance", ("runtime", "unit")),
    SelectorSupport("block_prefix", ("omitted", "output")),
    SelectorSupport("block_suffix", ("omitted", "output")),
)


_CUDAX_REDUCE = CppApi(
    headers=("cuda/experimental/coop.cuh", "cuda/experimental/group.cuh"),
    entity="::cuda::experimental::coop",
    methods=("reduce",),
    stability=ApiStability.EXPERIMENTAL,
)
_BLOCK_REDUCE = CppApi(
    headers=("cub/block/block_reduce.cuh",),
    entity="::cub::BlockReduce",
    methods=("Sum", "Reduce"),
    stability=ApiStability.PUBLIC,
)
_BLOCK_ADJACENT_DIFFERENCE = CppApi(
    headers=("cub/block/block_adjacent_difference.cuh",),
    entity="::cub::BlockAdjacentDifference",
    methods=(
        "SubtractLeft",
        "SubtractLeftPartialTile",
        "SubtractRight",
        "SubtractRightPartialTile",
    ),
    stability=ApiStability.PUBLIC,
)
_BLOCK_DISCONTINUITY = CppApi(
    headers=("cub/block/block_discontinuity.cuh",),
    entity="::cub::BlockDiscontinuity",
    methods=("FlagHeads", "FlagTails", "FlagHeadsAndTails"),
    stability=ApiStability.PUBLIC,
)
_BLOCK_EXCHANGE = CppApi(
    headers=("cub/block/block_exchange.cuh",),
    entity="::cub::BlockExchange",
    methods=(
        "StripedToBlocked",
        "BlockedToStriped",
        "WarpStripedToBlocked",
        "BlockedToWarpStriped",
        "ScatterToBlocked",
        "ScatterToStriped",
        "ScatterToStripedGuarded",
        "ScatterToStripedFlagged",
    ),
    stability=ApiStability.PUBLIC,
)
_BLOCK_EXCHANGE_GROUP = CppApi(
    headers=_BLOCK_EXCHANGE.headers,
    entity=_BLOCK_EXCHANGE.entity,
    methods=("StripedToBlocked", "BlockedToStriped"),
    stability=_BLOCK_EXCHANGE.stability,
)
_BLOCK_LOAD = CppApi(
    headers=("cub/block/block_load.cuh",),
    entity="::cub::BlockLoad",
    methods=("Load",),
    stability=ApiStability.PUBLIC,
)
_BLOCK_STORE = CppApi(
    headers=("cub/block/block_store.cuh",),
    entity="::cub::BlockStore",
    methods=("Store",),
    stability=ApiStability.PUBLIC,
)
_BLOCK_HISTOGRAM = CppApi(
    headers=("cub/block/block_histogram.cuh",),
    entity="::cub::BlockHistogram",
    methods=("Histogram",),
    stability=ApiStability.PUBLIC,
)
_BLOCK_MERGE_SORT = CppApi(
    headers=("cub/block/block_merge_sort.cuh",),
    entity="::cub::BlockMergeSort",
    methods=("Sort",),
    stability=ApiStability.PUBLIC,
)
_BLOCK_RADIX_RANK = CppApi(
    headers=("cub/block/block_radix_rank.cuh",),
    entity="::cub::BlockRadixRank",
    methods=("RankKeys",),
    stability=ApiStability.PUBLIC,
)
_BLOCK_RADIX_SORT = CppApi(
    headers=("cub/block/block_radix_sort.cuh",),
    entity="::cub::BlockRadixSort",
    methods=("Sort", "SortDescending"),
    stability=ApiStability.PUBLIC,
)
_BLOCK_ROW_REDUCE = CppApi(
    headers=("cub/block/block_row_reduce.cuh",),
    entity="::cub::BlockRowReduceWarpBroadcast",
    methods=("Sum",),
    stability=ApiStability.UNVERIFIED,
    availability=ApiAvailability.NOT_IN_TREE,
)
_BLOCK_RUN_LENGTH_DECODE = CppApi(
    headers=("cub/block/block_run_length_decode.cuh",),
    entity="::cub::BlockRunLengthDecode",
    methods=("RunLengthDecode",),
    stability=ApiStability.PUBLIC,
)
_BLOCK_SCAN = CppApi(
    headers=("cub/block/block_scan.cuh",),
    entity="::cub::BlockScan",
    methods=("ExclusiveSum", "ExclusiveScan", "InclusiveSum", "InclusiveScan"),
    stability=ApiStability.PUBLIC,
)
_BLOCK_SHUFFLE = CppApi(
    headers=("cub/block/block_shuffle.cuh",),
    entity="::cub::BlockShuffle",
    methods=("Up", "Down", "Offset", "Rotate"),
    stability=ApiStability.PUBLIC,
)
_BLOCK_TOPK_DETAIL = CppApi(
    headers=("cub/block/block_topk.cuh",),
    entity="::cub::detail::block_topk",
    methods=("max_keys", "min_keys", "max_pairs", "min_pairs"),
    stability=ApiStability.DETAIL,
)
_WARP_REDUCE = CppApi(
    headers=("cub/warp/warp_reduce.cuh",),
    entity="::cub::WarpReduce",
    methods=("Sum", "Reduce", "Min", "Max"),
    stability=ApiStability.PUBLIC,
)
_WARP_SCAN = CppApi(
    headers=("cub/warp/warp_scan.cuh",),
    entity="::cub::WarpScan",
    methods=(
        "ExclusiveSum",
        "ExclusiveScan",
        "ExclusiveScanPartial",
        "InclusiveSum",
        "InclusiveScan",
        "InclusiveScanPartial",
    ),
    stability=ApiStability.PUBLIC,
)
_WARP_SCAN_GROUP = CppApi(
    headers=_WARP_SCAN.headers,
    entity=_WARP_SCAN.entity,
    methods=("ExclusiveSum", "ExclusiveScan", "InclusiveSum", "InclusiveScan"),
    stability=_WARP_SCAN.stability,
)
_WARP_EXCHANGE = CppApi(
    headers=("cub/warp/warp_exchange.cuh",),
    entity="::cub::WarpExchange",
    methods=("StripedToBlocked", "BlockedToStriped", "ScatterToStriped"),
    stability=ApiStability.PUBLIC,
)
_WARP_EXCHANGE_GROUP = CppApi(
    headers=_WARP_EXCHANGE.headers,
    entity=_WARP_EXCHANGE.entity,
    methods=("StripedToBlocked", "BlockedToStriped"),
    stability=_WARP_EXCHANGE.stability,
)
_WARP_LOAD = CppApi(
    headers=("cub/warp/warp_load.cuh",),
    entity="::cub::WarpLoad",
    methods=("Load",),
    stability=ApiStability.PUBLIC,
)
_WARP_STORE = CppApi(
    headers=("cub/warp/warp_store.cuh",),
    entity="::cub::WarpStore",
    methods=("Store",),
    stability=ApiStability.PUBLIC,
)
_WARP_MERGE_SORT = CppApi(
    headers=("cub/warp/warp_merge_sort.cuh",),
    entity="::cub::WarpMergeSort",
    methods=("Sort",),
    stability=ApiStability.PUBLIC,
)


def _generated(note: str) -> ProviderProvenance:
    return ProviderProvenance(ProvenanceKind.GENERATED_HANDWRITTEN, note)


def _cub(api: CppApi, note: str) -> ProviderProvenance:
    return ProviderProvenance(ProvenanceKind.CUB_PUBLIC, note, api)


def _payload_adapter(name: str, note: str) -> PayloadAdapterProvenance:
    return PayloadAdapterProvenance(
        name,
        ProviderProvenance(ProvenanceKind.PAYLOAD_ADAPTER, note),
    )


_CUDAX_GROUP_REDUCE = ProviderProvenance(
    ProvenanceKind.CUDAX_PUBLIC,
    "the group-first renderer calls the official broadcasted or root-only "
    "CUDAX reduction",
    _CUDAX_REDUCE,
)
_CUDAX_SCOPED_REDUCE = ProviderProvenance(
    ProvenanceKind.CUDAX_PUBLIC,
    "the builtin scoped full-block and physical-warp defaults delegate to the "
    "shared broadcasted CUDAX reduction artifact",
    _CUDAX_REDUCE,
)
_CUB_BLOCK_REDUCE = _cub(
    _BLOCK_REDUCE,
    "the selected block variant calls the public CUB BlockReduce API",
)
_CUB_PHYSICAL_WARP_REDUCE = _cub(
    _WARP_REDUCE,
    "the selected physical-warp variant calls the public CUB WarpReduce API",
)
_CUB_LOGICAL_WARP_REDUCE = ProviderProvenance(
    ProvenanceKind.CUB_PUBLIC_WITH_GENERATED_ADAPTER,
    "the logical-warp scoped provider calls public WarpReduce after optional "
    "generated per-thread folding",
    _WARP_REDUCE,
)
_GROUP_REDUCE_RESULT_SELECTOR = SelectorSupport(
    "broadcast",
    ("true", "false"),
)
_GROUP_REDUCE_VALID_SELECTOR = SelectorSupport(
    "valid_items",
    ("static", "runtime"),
)
_GROUP_BLOCK_REDUCE_ALGORITHM_SELECTOR = SelectorSupport(
    "algorithm",
    ("raking_commutative_only", "raking", "warp_reductions"),
)
_GROUP_BLOCK_REDUCE_ROUTES = (
    AlternateLoweringRoute(
        name="direct_cub",
        cases=(
            AlternateLoweringRouteCase(
                name="partial_scalar",
                operand_kinds=(OperandKind.SCALAR,),
                selector_values=(
                    ("broadcast", ("false",)),
                    ("valid_items", ("static", "runtime")),
                    (
                        "algorithm",
                        (
                            "omitted",
                            "raking",
                            "warp_reductions",
                        ),
                    ),
                ),
            ),
            AlternateLoweringRouteCase(
                name="partial_scalar_commutative",
                operand_kinds=(OperandKind.SCALAR,),
                selector_values=(
                    ("broadcast", ("false",)),
                    ("valid_items", ("static", "runtime")),
                    ("algorithm", ("raking_commutative_only",)),
                ),
                requires_commutative_operator=True,
            ),
            AlternateLoweringRouteCase(
                name="explicit_algorithm",
                operand_kinds=(OperandKind.SCALAR, OperandKind.THREAD_DATA),
                selector_values=(
                    ("broadcast", ("false",)),
                    ("valid_items", ("omitted",)),
                    (
                        "algorithm",
                        (
                            "raking",
                            "warp_reductions",
                        ),
                    ),
                ),
            ),
            AlternateLoweringRouteCase(
                name="explicit_commutative_algorithm",
                operand_kinds=(OperandKind.SCALAR, OperandKind.THREAD_DATA),
                selector_values=(
                    ("broadcast", ("false",)),
                    ("valid_items", ("omitted",)),
                    ("algorithm", ("raking_commutative_only",)),
                ),
                requires_commutative_operator=True,
            ),
        ),
        target=GroupLoweringTarget.CUB_BLOCK,
        api=_BLOCK_REDUCE,
        provenance=_CUB_BLOCK_REDUCE,
    ),
)
_GROUP_WARP_REDUCE_ROUTES = (
    AlternateLoweringRoute(
        name="direct_cub_partial",
        cases=(
            AlternateLoweringRouteCase(
                name="partial_scalar",
                operand_kinds=(OperandKind.SCALAR,),
                selector_values=(
                    ("broadcast", ("false",)),
                    ("valid_items", ("static", "runtime")),
                ),
            ),
        ),
        target=GroupLoweringTarget.CUB_WARP,
        api=_WARP_REDUCE,
        provenance=_CUB_PHYSICAL_WARP_REDUCE,
    ),
)
_SCOPED_BLOCK_REDUCE_ROUTES = (
    AlternateLoweringRoute(
        name="direct_cub",
        cases=(
            AlternateLoweringRouteCase(
                name="partial_scalar",
                operand_kinds=(OperandKind.SCALAR,),
                selector_values=(
                    ("valid_items", ("static", "runtime")),
                    (
                        "algorithm",
                        (
                            "omitted",
                            "raking",
                            "warp_reductions",
                        ),
                    ),
                ),
            ),
            AlternateLoweringRouteCase(
                name="partial_scalar_commutative",
                operand_kinds=(OperandKind.SCALAR,),
                selector_values=(
                    ("valid_items", ("static", "runtime")),
                    ("algorithm", ("raking_commutative_only",)),
                ),
                requires_commutative_operator=True,
            ),
            AlternateLoweringRouteCase(
                name="explicit_algorithm",
                operand_kinds=(OperandKind.SCALAR, OperandKind.THREAD_DATA),
                selector_values=(
                    ("valid_items", ("omitted",)),
                    ("algorithm", ("raking", "warp_reductions")),
                ),
            ),
            AlternateLoweringRouteCase(
                name="explicit_commutative_algorithm",
                operand_kinds=(OperandKind.SCALAR, OperandKind.THREAD_DATA),
                selector_values=(
                    ("valid_items", ("omitted",)),
                    ("algorithm", ("raking_commutative_only",)),
                ),
                requires_commutative_operator=True,
            ),
        ),
        target=GroupLoweringTarget.CUB_BLOCK,
        api=_BLOCK_REDUCE,
        provenance=_CUB_BLOCK_REDUCE,
    ),
)
_SCOPED_WARP_REDUCE_WIDTH_SELECTOR = SelectorSupport(
    "threads_in_warp",
    ("physical", "logical"),
)
_SCOPED_WARP_REDUCE_ROUTES = (
    AlternateLoweringRoute(
        name="direct_cub_physical_partial",
        cases=(
            AlternateLoweringRouteCase(
                name="physical_partial_scalar",
                operand_kinds=(OperandKind.SCALAR,),
                selector_values=(
                    ("threads_in_warp", ("physical",)),
                    ("valid_items", ("static", "runtime")),
                ),
            ),
        ),
        target=GroupLoweringTarget.CUB_WARP,
        api=_WARP_REDUCE,
        provenance=_CUB_PHYSICAL_WARP_REDUCE,
    ),
    AlternateLoweringRoute(
        name="logical_warp_cub_adapter",
        cases=(
            AlternateLoweringRouteCase(
                name="logical_full",
                operand_kinds=(OperandKind.SCALAR, OperandKind.THREAD_DATA),
                selector_values=(
                    ("threads_in_warp", ("logical",)),
                    ("valid_items", ("omitted",)),
                ),
            ),
            AlternateLoweringRouteCase(
                name="logical_partial",
                operand_kinds=(OperandKind.SCALAR, OperandKind.THREAD_DATA),
                selector_values=(
                    ("threads_in_warp", ("logical",)),
                    ("valid_items", ("static", "runtime")),
                ),
            ),
        ),
        target=GroupLoweringTarget.CUB_WARP,
        api=_WARP_REDUCE,
        provenance=_CUB_LOGICAL_WARP_REDUCE,
    ),
)

_CUB_BLOCK_EXCHANGE = _cub(
    _BLOCK_EXCHANGE,
    "all builtin scoped block modes call the public CUB BlockExchange API "
    "through one whole-register-array provider",
)
_CUB_BLOCK_ADJACENT_DIFFERENCE = _cub(
    _BLOCK_ADJACENT_DIFFERENCE,
    "the root and builtin scoped block paths share one typed provider that "
    "calls the public CUB BlockAdjacentDifference API",
)
_CUB_BLOCK_DISCONTINUITY = _cub(
    _BLOCK_DISCONTINUITY,
    "the root and builtin scoped block paths share one typed provider that "
    "calls one public CUB BlockDiscontinuity method",
)
_CUB_BLOCK_SHUFFLE = _cub(
    _BLOCK_SHUFFLE,
    "the root and builtin scoped representable routes share one typed provider "
    "that calls the public CUB BlockShuffle API",
)
_CUB_BLOCK_RADIX_RANK = _cub(
    _BLOCK_RADIX_RANK,
    "the root and builtin scoped block paths share one typed provider that "
    "calls the public CUB BlockRadixRank API",
)
_CUB_BLOCK_RADIX_SORT = _cub(
    _BLOCK_RADIX_SORT,
    "the root and builtin scoped key and pair paths share one typed provider "
    "that calls the public CUB BlockRadixSort API",
)
_CUB_BLOCK_MERGE_SORT = _cub(
    _BLOCK_MERGE_SORT,
    "the root and builtin scoped block paths share one typed provider that "
    "invokes public CUB BlockMergeSort exactly once for keys or pairs",
)
_CUB_PHYSICAL_WARP_MERGE_SORT = _cub(
    _WARP_MERGE_SORT,
    "the root and builtin scoped physical-warp paths share one typed provider "
    "that invokes public CUB WarpMergeSort exactly once for keys or pairs",
)
_CUB_LOGICAL_WARP_MERGE_SORT = ProviderProvenance(
    ProvenanceKind.CUB_PUBLIC_WITH_GENERATED_ADAPTER,
    "the scoped logical-warp path selects one public WarpMergeSort TempStorage "
    "instance per complete logical warp",
    _WARP_MERGE_SORT,
)
_SCOPED_WARP_MERGE_SORT_WIDTH_SELECTOR = SelectorSupport(
    "threads_in_warp",
    ("physical", "logical"),
)
_SCOPED_WARP_MERGE_SORT_ROUTES = (
    AlternateLoweringRoute(
        name="logical_warp_cub_adapter",
        cases=(
            AlternateLoweringRouteCase(
                name="logical_full_or_partial",
                operand_kinds=(OperandKind.THREAD_DATA,),
                selector_values=(("threads_in_warp", ("logical",)),),
            ),
        ),
        target=GroupLoweringTarget.CUB_WARP,
        api=_WARP_MERGE_SORT,
        provenance=_CUB_LOGICAL_WARP_MERGE_SORT,
    ),
)
_CUB_BLOCK_EXCHANGE_GROUP = _cub(
    _BLOCK_EXCHANGE_GROUP,
    "the root and overlapping builtin scoped block modes share one canonical "
    "provider that calls the public CUB BlockExchange API",
)
_CUB_PHYSICAL_WARP_EXCHANGE = _cub(
    _WARP_EXCHANGE_GROUP,
    "the root and builtin scoped full physical-warp modes share one canonical "
    "provider that calls the public CUB WarpExchange API",
)
_CUB_WARP_EXCHANGE_COMPATIBILITY = _cub(
    _WARP_EXCHANGE,
    "logical-warp and scatter scoped modes call the public CUB WarpExchange "
    "API through the compatibility provider",
)
_CUB_BLOCK_LOAD = _cub(
    _BLOCK_LOAD,
    "the group-first provider calls public CUB BlockLoad with a raw contiguous "
    "tensor pointer",
)
_CUB_BLOCK_STORE = _cub(
    _BLOCK_STORE,
    "the group-first provider calls public CUB BlockStore with a raw contiguous "
    "tensor pointer",
)
_CUB_BLOCK_HISTOGRAM = _cub(
    _BLOCK_HISTOGRAM,
    "the root and builtin scoped block paths share one static-width provider "
    "that calls public CUB BlockHistogram and projects its shared counters",
)
_CUB_BLOCK_RUN_LENGTH_DECODE = _cub(
    _BLOCK_RUN_LENGTH_DECODE,
    "the root and builtin scoped block paths share one fused provider over "
    "the public CUB BlockRunLengthDecode lifecycle",
)
_CUB_PHYSICAL_WARP_LOAD = _cub(
    _WARP_LOAD,
    "the group-first provider calls public CUB WarpLoad and offsets each "
    "physical warp to its own tile",
)
_CUB_PHYSICAL_WARP_STORE = _cub(
    _WARP_STORE,
    "the group-first provider calls public CUB WarpStore and offsets each "
    "physical warp to its own tile",
)
_CUTE_LOAD_STORE_PAYLOAD_ADAPTER = _payload_adapter(
    "cute_indexing",
    "non-contiguous, statically unproven, or CUB-incompatible CuTe tensor "
    "operands retain the explicit direct/striped indexing payload adapter "
    "without claiming CUB",
)
_PRIMS_LOAD_STORE_PAYLOAD_ADAPTER = _payload_adapter(
    "prims_array",
    "cutlass.Array values and Prims-specific bounds or memory controls retain "
    "the Prims array payload adapter without claiming a C++ collective",
)
_LOAD_STORE_PAYLOAD_ADAPTERS = (
    _CUTE_LOAD_STORE_PAYLOAD_ADAPTER,
    _PRIMS_LOAD_STORE_PAYLOAD_ADAPTER,
)
_GROUP_LOAD_STORE_VALID_SELECTOR = SelectorSupport(
    "valid_items",
    ("static", "runtime"),
)
_GROUP_LOAD_OOB_SELECTOR = SelectorSupport(
    "oob_default",
    ("static", "runtime"),
)
_GROUP_LOAD_STORE_OFFSET_SELECTOR = SelectorSupport(
    "offset",
    ("static", "runtime"),
)
_GROUP_BLOCK_LOAD_STORE_ALGORITHM_SELECTOR = SelectorSupport(
    "algorithm",
    (
        "direct",
        "striped",
        "vectorize",
        "transpose",
        "warp_transpose",
        "warp_transpose_timesliced",
    ),
)
_GROUP_WARP_LOAD_STORE_ALGORITHM_SELECTOR = SelectorSupport(
    "algorithm",
    ("direct", "striped", "vectorize", "transpose"),
)
_SCOPED_BLOCK_EXCHANGE_MODE_SELECTOR = SelectorSupport(
    "mode",
    (
        "striped_to_blocked",
        "blocked_to_striped",
        "warp_striped_to_blocked",
        "blocked_to_warp_striped",
        "scatter_to_blocked",
        "scatter_to_striped",
        "scatter_to_striped_guarded",
        "scatter_to_striped_flagged",
    ),
)
_SCOPED_BLOCK_EXCHANGE_ROUTES = (
    AlternateLoweringRoute(
        name="direct_cub_extended_modes",
        cases=(
            AlternateLoweringRouteCase(
                name="extended_thread_data",
                operand_kinds=(OperandKind.THREAD_DATA,),
                selector_values=(
                    (
                        "mode",
                        (
                            "warp_striped_to_blocked",
                            "blocked_to_warp_striped",
                            "scatter_to_blocked",
                            "scatter_to_striped",
                            "scatter_to_striped_guarded",
                            "scatter_to_striped_flagged",
                        ),
                    ),
                ),
                min_items_per_thread=1,
                max_items_per_thread=MAX_GROUP_EXCHANGE_ITEMS_PER_THREAD,
            ),
        ),
        target=GroupLoweringTarget.CUB_BLOCK,
        api=_BLOCK_EXCHANGE,
        provenance=_CUB_BLOCK_EXCHANGE,
    ),
)
_SCOPED_WARP_EXCHANGE_WIDTH_SELECTOR = SelectorSupport(
    "threads_in_warp",
    ("physical", "logical"),
)
_SCOPED_WARP_EXCHANGE_MODE_SELECTOR = SelectorSupport(
    "mode",
    ("striped_to_blocked", "blocked_to_striped", "scatter_to_striped"),
)
_SCOPED_WARP_EXCHANGE_ROUTES = (
    AlternateLoweringRoute(
        name="logical_warp_x4_compatibility",
        cases=(
            AlternateLoweringRouteCase(
                name="logical_common_mode_x4",
                operand_kinds=(OperandKind.THREAD_DATA,),
                selector_values=(
                    ("threads_in_warp", ("logical",)),
                    ("mode", ("striped_to_blocked", "blocked_to_striped")),
                ),
                min_items_per_thread=1,
                max_items_per_thread=4,
            ),
        ),
        target=GroupLoweringTarget.CUB_WARP,
        api=_WARP_EXCHANGE,
        provenance=_CUB_WARP_EXCHANGE_COMPATIBILITY,
    ),
    AlternateLoweringRoute(
        name="scatter_x4_compatibility",
        cases=(
            AlternateLoweringRouteCase(
                name="physical_or_logical_scatter_x4",
                operand_kinds=(OperandKind.THREAD_DATA,),
                selector_values=(
                    ("threads_in_warp", ("physical", "logical")),
                    ("mode", ("scatter_to_striped",)),
                ),
                min_items_per_thread=1,
                max_items_per_thread=4,
            ),
        ),
        target=GroupLoweringTarget.CUB_WARP,
        api=_WARP_EXCHANGE,
        provenance=_CUB_WARP_EXCHANGE_COMPATIBILITY,
    ),
)

_CUB_BLOCK_SCAN = _cub(
    _BLOCK_SCAN,
    "the root and builtin scoped block paths share one canonical provider "
    "that calls the public CUB BlockScan API",
)
_CUB_PHYSICAL_WARP_SCAN = _cub(
    _WARP_SCAN_GROUP,
    "the root and builtin scoped full physical-warp scalar paths share one "
    "canonical provider that calls the public CUB WarpScan API",
)
_CUB_WARP_SCAN_SCALAR_COMPATIBILITY = _cub(
    _WARP_SCAN,
    "logical-warp or valid-item scoped scalar paths call the public CUB "
    "WarpScan API through the legacy provider",
)
_CUB_WARP_SCAN_THREAD_DATA_COMPATIBILITY = ProviderProvenance(
    ProvenanceKind.CUB_PUBLIC_WITH_GENERATED_ADAPTER,
    "scoped ThreadData paths use the legacy generated per-lane folding and "
    "prefix-reconstruction adapter around public WarpScan",
    _WARP_SCAN,
)
_GROUP_BLOCK_SCAN_ALGORITHM_SELECTOR = SelectorSupport(
    "algorithm",
    ("raking", "raking_memoize", "warp_scans"),
)
_SCOPED_WARP_SCAN_WIDTH_SELECTOR = SelectorSupport(
    "threads_in_warp",
    ("physical", "logical"),
)
_SCOPED_WARP_SCAN_VALID_SELECTOR = SelectorSupport(
    "valid_items",
    ("static", "runtime"),
)
_SCOPED_WARP_SCAN_ROUTES = (
    AlternateLoweringRoute(
        name="direct_cub_scalar_compatibility",
        cases=(
            AlternateLoweringRouteCase(
                name="physical_partial_scalar",
                operand_kinds=(OperandKind.SCALAR,),
                selector_values=(
                    ("threads_in_warp", ("physical",)),
                    ("valid_items", ("static", "runtime")),
                ),
            ),
            AlternateLoweringRouteCase(
                name="logical_full_scalar",
                operand_kinds=(OperandKind.SCALAR,),
                selector_values=(
                    ("threads_in_warp", ("logical",)),
                    ("valid_items", ("omitted",)),
                ),
            ),
            AlternateLoweringRouteCase(
                name="logical_partial_scalar",
                operand_kinds=(OperandKind.SCALAR,),
                selector_values=(
                    ("threads_in_warp", ("logical",)),
                    ("valid_items", ("static", "runtime")),
                ),
            ),
        ),
        target=GroupLoweringTarget.CUB_WARP,
        api=_WARP_SCAN,
        provenance=_CUB_WARP_SCAN_SCALAR_COMPATIBILITY,
    ),
    AlternateLoweringRoute(
        name="thread_data_cub_adapter",
        cases=(
            AlternateLoweringRouteCase(
                name="physical_thread_data_full",
                operand_kinds=(OperandKind.THREAD_DATA,),
                selector_values=(
                    ("threads_in_warp", ("physical",)),
                    ("valid_items", ("omitted",)),
                ),
            ),
            AlternateLoweringRouteCase(
                name="physical_thread_data_partial",
                operand_kinds=(OperandKind.THREAD_DATA,),
                selector_values=(
                    ("threads_in_warp", ("physical",)),
                    ("valid_items", ("static", "runtime")),
                ),
            ),
            AlternateLoweringRouteCase(
                name="logical_thread_data_full",
                operand_kinds=(OperandKind.THREAD_DATA,),
                selector_values=(
                    ("threads_in_warp", ("logical",)),
                    ("valid_items", ("omitted",)),
                ),
            ),
            AlternateLoweringRouteCase(
                name="logical_thread_data_partial",
                operand_kinds=(OperandKind.THREAD_DATA,),
                selector_values=(
                    ("threads_in_warp", ("logical",)),
                    ("valid_items", ("static", "runtime")),
                ),
            ),
        ),
        target=GroupLoweringTarget.CUB_WARP,
        api=_WARP_SCAN,
        provenance=_CUB_WARP_SCAN_THREAD_DATA_COMPATIBILITY,
    ),
)


CAPABILITIES = (
    Capability(
        OperationFamily.ADJACENT_DIFFERENCE,
        GroupKind.BLOCK,
        (_SCALAR, _THREAD_DATA),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_ADJACENT_DIFFERENCE,
        _CUB_BLOCK_ADJACENT_DIFFERENCE,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_BLOCK_ADJACENT_DIFFERENCE,
        group_first_selector_support=_GROUP_ADJACENT_DIFFERENCE_SELECTORS,
        builtin_scoped_selector_support=_GROUP_ADJACENT_DIFFERENCE_SELECTORS,
    ),
    Capability(
        OperationFamily.DISCONTINUITY,
        GroupKind.BLOCK,
        (_SCALAR, _THREAD_DATA),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_DISCONTINUITY,
        _CUB_BLOCK_DISCONTINUITY,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_BLOCK_DISCONTINUITY,
        group_first_selector_support=_GROUP_DISCONTINUITY_SELECTORS,
        builtin_scoped_selector_support=_GROUP_DISCONTINUITY_SELECTORS,
    ),
    Capability(
        OperationFamily.EXCHANGE,
        GroupKind.BLOCK,
        (_THREAD_DATA_EXCHANGE,),
        (_THREAD_DATA_EXCHANGE,),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_EXCHANGE_GROUP,
        _CUB_BLOCK_EXCHANGE_GROUP,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_BLOCK_EXCHANGE_GROUP,
        group_first_selector_support=_GROUP_EXCHANGE_MODES,
        builtin_scoped_selector_support=(_SCOPED_BLOCK_EXCHANGE_MODE_SELECTOR,),
        alternate_builtin_scoped_routes=_SCOPED_BLOCK_EXCHANGE_ROUTES,
    ),
    Capability(
        OperationFamily.HISTOGRAM,
        GroupKind.BLOCK,
        (_SCALAR, _THREAD_DATA),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_HISTOGRAM,
        _CUB_BLOCK_HISTOGRAM,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_BLOCK_HISTOGRAM,
    ),
    Capability(
        OperationFamily.LOAD,
        GroupKind.BLOCK,
        (_TENSOR,),
        (_TENSOR,),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_LOAD,
        _CUB_BLOCK_LOAD,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_BLOCK_LOAD,
        group_first_selector_support=(
            _GROUP_BLOCK_LOAD_STORE_ALGORITHM_SELECTOR,
            _GROUP_LOAD_STORE_VALID_SELECTOR,
            _GROUP_LOAD_OOB_SELECTOR,
            _GROUP_LOAD_STORE_OFFSET_SELECTOR,
        ),
        scoped_payload_adapter_provenance=_LOAD_STORE_PAYLOAD_ADAPTERS,
    ),
    Capability(
        OperationFamily.MERGE_SORT,
        GroupKind.BLOCK,
        (_SCALAR, _THREAD_DATA),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_MERGE_SORT,
        _CUB_BLOCK_MERGE_SORT,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_BLOCK_MERGE_SORT,
    ),
    Capability(
        OperationFamily.RADIX_RANK,
        GroupKind.BLOCK,
        (_SCALAR, _THREAD_DATA),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_RADIX_RANK,
        _CUB_BLOCK_RADIX_RANK,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_BLOCK_RADIX_RANK,
    ),
    Capability(
        OperationFamily.RADIX_SORT,
        GroupKind.BLOCK,
        (_SCALAR, _THREAD_DATA),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_RADIX_SORT,
        _CUB_BLOCK_RADIX_SORT,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_BLOCK_RADIX_SORT,
    ),
    Capability(
        OperationFamily.REDUCE,
        GroupKind.BLOCK,
        (_SCALAR, _THREAD_DATA),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUDAX_GROUP,
        _CUDAX_REDUCE,
        _CUDAX_SCOPED_REDUCE,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUDAX_GROUP_REDUCE,
        group_first_selector_support=(
            _GROUP_REDUCE_RESULT_SELECTOR,
            _GROUP_REDUCE_VALID_SELECTOR,
            _GROUP_BLOCK_REDUCE_ALGORITHM_SELECTOR,
        ),
        alternate_group_first_routes=_GROUP_BLOCK_REDUCE_ROUTES,
        builtin_scoped_selector_support=(
            _GROUP_REDUCE_VALID_SELECTOR,
            _GROUP_BLOCK_REDUCE_ALGORITHM_SELECTOR,
        ),
        alternate_builtin_scoped_routes=_SCOPED_BLOCK_REDUCE_ROUTES,
    ),
    Capability(
        OperationFamily.REDUCE,
        GroupKind.THREAD,
        (),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUDAX_GROUP,
        _CUDAX_REDUCE,
        None,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUDAX_GROUP_REDUCE,
        group_first_selector_support=(_GROUP_REDUCE_RESULT_SELECTOR,),
    ),
    Capability(
        OperationFamily.REDUCE,
        GroupKind.THREADS_WITHIN_WARP,
        (),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUDAX_GROUP,
        _CUDAX_REDUCE,
        None,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUDAX_GROUP_REDUCE,
        group_first_selector_support=(_GROUP_REDUCE_RESULT_SELECTOR,),
    ),
    Capability(
        OperationFamily.REDUCE,
        GroupKind.WARPS_WITHIN_BLOCK,
        (),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUDAX_GROUP,
        _CUDAX_REDUCE,
        None,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUDAX_GROUP_REDUCE,
        group_first_selector_support=(_GROUP_REDUCE_RESULT_SELECTOR,),
    ),
    Capability(
        OperationFamily.REDUCE,
        GroupKind.CLUSTER,
        (),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUDAX_GROUP,
        _CUDAX_REDUCE,
        None,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUDAX_GROUP_REDUCE,
        group_first_selector_support=(_GROUP_REDUCE_RESULT_SELECTOR,),
    ),
    Capability(
        OperationFamily.REDUCE,
        GroupKind.GRID,
        (),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUDAX_GROUP,
        _CUDAX_REDUCE,
        None,
        GroupFirstReadiness.BLOCKED_PROVIDER_CONVERSION,
        "grid Reduce is blocked until the CUTLASS DSL provides a reviewed "
        "compiler-managed device workspace contract",
        group_first_selector_support=(_GROUP_REDUCE_RESULT_SELECTOR,),
    ),
    Capability(
        OperationFamily.ROW_REDUCE,
        GroupKind.BLOCK,
        (_SCALAR,),
        (),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_ROW_REDUCE,
        ProviderProvenance(
            ProvenanceKind.CUB_NOT_IN_TREE,
            "when the non-tree CUB header is supplied, the scoped provider calls "
            "BlockRowReduceWarpBroadcast::Sum",
            _BLOCK_ROW_REDUCE,
        ),
        GroupFirstReadiness.BLOCKED_PLANNER_AND_DEPENDENCY,
        "group-first row reduction is not modeled and its CUB header is not "
        "present in this source tree",
    ),
    Capability(
        OperationFamily.RUN_LENGTH_DECODE,
        GroupKind.BLOCK,
        (_SCALAR, _THREAD_DATA),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_RUN_LENGTH_DECODE,
        _CUB_BLOCK_RUN_LENGTH_DECODE,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_BLOCK_RUN_LENGTH_DECODE,
    ),
    Capability(
        OperationFamily.SCAN,
        GroupKind.BLOCK,
        (_SCALAR, _THREAD_DATA),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_SCAN,
        _CUB_BLOCK_SCAN,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_BLOCK_SCAN,
        group_first_selector_support=(_GROUP_BLOCK_SCAN_ALGORITHM_SELECTOR,),
    ),
    Capability(
        OperationFamily.SHUFFLE,
        GroupKind.BLOCK,
        (_SHUFFLE_SCALAR, _SHUFFLE_THREAD_DATA),
        (_SHUFFLE_SCALAR, _SHUFFLE_THREAD_DATA),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_SHUFFLE,
        _CUB_BLOCK_SHUFFLE,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_BLOCK_SHUFFLE,
        group_first_selector_support=_GROUP_SHUFFLE_SELECTORS,
        builtin_scoped_selector_support=_GROUP_SHUFFLE_SELECTORS,
    ),
    Capability(
        OperationFamily.STORE,
        GroupKind.BLOCK,
        (_TENSOR, _THREAD_DATA),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_STORE,
        _CUB_BLOCK_STORE,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_BLOCK_STORE,
        group_first_selector_support=(
            _GROUP_BLOCK_LOAD_STORE_ALGORITHM_SELECTOR,
            _GROUP_LOAD_STORE_VALID_SELECTOR,
            _GROUP_LOAD_STORE_OFFSET_SELECTOR,
        ),
        scoped_payload_adapter_provenance=_LOAD_STORE_PAYLOAD_ADAPTERS,
    ),
    Capability(
        OperationFamily.TOPK,
        GroupKind.BLOCK,
        (_SCALAR, _THREAD_DATA),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUB_BLOCK,
        _BLOCK_TOPK_DETAIL,
        ProviderProvenance(
            ProvenanceKind.CUB_DETAIL,
            "the scoped provider uses the undocumented detail::block_topk type",
            _BLOCK_TOPK_DETAIL,
        ),
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=ProviderProvenance(
            ProvenanceKind.CUB_DETAIL,
            "the certified keys-only group adapter delegates to the pinned "
            "detail::block_topk provider",
            _BLOCK_TOPK_DETAIL,
        ),
    ),
    Capability(
        OperationFamily.EXCHANGE,
        GroupKind.WARP,
        (_THREAD_DATA_EXCHANGE,),
        (_THREAD_DATA_EXCHANGE,),
        GroupLoweringTarget.CUB_WARP,
        _WARP_EXCHANGE_GROUP,
        _CUB_PHYSICAL_WARP_EXCHANGE,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_PHYSICAL_WARP_EXCHANGE,
        group_first_selector_support=_GROUP_EXCHANGE_MODES,
        builtin_scoped_selector_support=(
            _SCOPED_WARP_EXCHANGE_WIDTH_SELECTOR,
            _SCOPED_WARP_EXCHANGE_MODE_SELECTOR,
        ),
        alternate_builtin_scoped_routes=_SCOPED_WARP_EXCHANGE_ROUTES,
    ),
    Capability(
        OperationFamily.LOAD,
        GroupKind.WARP,
        (_TENSOR,),
        (_TENSOR,),
        GroupLoweringTarget.CUB_WARP,
        _WARP_LOAD,
        _CUB_PHYSICAL_WARP_LOAD,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_PHYSICAL_WARP_LOAD,
        group_first_selector_support=(
            _GROUP_WARP_LOAD_STORE_ALGORITHM_SELECTOR,
            _GROUP_LOAD_STORE_VALID_SELECTOR,
            _GROUP_LOAD_OOB_SELECTOR,
            _GROUP_LOAD_STORE_OFFSET_SELECTOR,
        ),
        scoped_payload_adapter_provenance=_LOAD_STORE_PAYLOAD_ADAPTERS,
    ),
    Capability(
        OperationFamily.MERGE_SORT,
        GroupKind.WARP,
        (_THREAD_DATA,),
        (_THREAD_DATA,),
        GroupLoweringTarget.CUB_WARP,
        _WARP_MERGE_SORT,
        _CUB_PHYSICAL_WARP_MERGE_SORT,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_PHYSICAL_WARP_MERGE_SORT,
        builtin_scoped_selector_support=(_SCOPED_WARP_MERGE_SORT_WIDTH_SELECTOR,),
        alternate_builtin_scoped_routes=_SCOPED_WARP_MERGE_SORT_ROUTES,
    ),
    Capability(
        OperationFamily.MERGE_SORT,
        GroupKind.THREADS_WITHIN_WARP,
        (),
        (_THREAD_DATA,),
        GroupLoweringTarget.CUB_WARP,
        _WARP_MERGE_SORT,
        None,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_LOGICAL_WARP_MERGE_SORT,
    ),
    Capability(
        OperationFamily.REDUCE,
        GroupKind.WARP,
        (_SCALAR, _THREAD_DATA),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUDAX_GROUP,
        _CUDAX_REDUCE,
        _CUDAX_SCOPED_REDUCE,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUDAX_GROUP_REDUCE,
        group_first_selector_support=(
            _GROUP_REDUCE_RESULT_SELECTOR,
            _GROUP_REDUCE_VALID_SELECTOR,
        ),
        alternate_group_first_routes=_GROUP_WARP_REDUCE_ROUTES,
        builtin_scoped_selector_support=(
            _SCOPED_WARP_REDUCE_WIDTH_SELECTOR,
            _GROUP_REDUCE_VALID_SELECTOR,
        ),
        alternate_builtin_scoped_routes=_SCOPED_WARP_REDUCE_ROUTES,
    ),
    Capability(
        OperationFamily.SCAN,
        GroupKind.WARP,
        (_SCALAR, _THREAD_DATA),
        (_SCALAR,),
        GroupLoweringTarget.CUB_WARP,
        _WARP_SCAN_GROUP,
        _CUB_PHYSICAL_WARP_SCAN,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_PHYSICAL_WARP_SCAN,
        builtin_scoped_selector_support=(
            _SCOPED_WARP_SCAN_WIDTH_SELECTOR,
            _SCOPED_WARP_SCAN_VALID_SELECTOR,
        ),
        alternate_builtin_scoped_routes=_SCOPED_WARP_SCAN_ROUTES,
    ),
    Capability(
        OperationFamily.STORE,
        GroupKind.WARP,
        (_TENSOR, _THREAD_DATA),
        (_SCALAR, _THREAD_DATA),
        GroupLoweringTarget.CUB_WARP,
        _WARP_STORE,
        _CUB_PHYSICAL_WARP_STORE,
        GroupFirstReadiness.READY,
        None,
        group_first_provenance=_CUB_PHYSICAL_WARP_STORE,
        group_first_selector_support=(
            _GROUP_WARP_LOAD_STORE_ALGORITHM_SELECTOR,
            _GROUP_LOAD_STORE_VALID_SELECTOR,
            _GROUP_LOAD_STORE_OFFSET_SELECTOR,
        ),
        scoped_payload_adapter_provenance=_LOAD_STORE_PAYLOAD_ADAPTERS,
    ),
)


_ALL_GROUP_QUERY_LEVELS = ("thread", "warp", "block", "cluster", "grid")
_GROUP_METHOD_VALIDATION_EVIDENCE = (
    "tests/backends/cutlass/runtime/test_group_hierarchy.py and "
    "tests/backends/cutlass/compile/test_cudax_group_plumbing.py"
)

GROUP_METHOD_CAPABILITIES = (
    GroupMethodCapability(
        GroupKind.THREAD,
        _ALL_GROUP_QUERY_LEVELS,
        True,
        True,
        GroupFirstReadiness.READY,
        validation_evidence=_GROUP_METHOD_VALIDATION_EVIDENCE,
    ),
    GroupMethodCapability(
        GroupKind.WARP,
        _ALL_GROUP_QUERY_LEVELS,
        True,
        True,
        GroupFirstReadiness.READY,
        validation_evidence=_GROUP_METHOD_VALIDATION_EVIDENCE,
    ),
    GroupMethodCapability(
        GroupKind.BLOCK,
        _ALL_GROUP_QUERY_LEVELS,
        True,
        True,
        GroupFirstReadiness.READY,
        validation_evidence=_GROUP_METHOD_VALIDATION_EVIDENCE,
    ),
    GroupMethodCapability(
        GroupKind.THREADS_WITHIN_WARP,
        _ALL_GROUP_QUERY_LEVELS,
        True,
        True,
        GroupFirstReadiness.READY,
        validation_evidence=_GROUP_METHOD_VALIDATION_EVIDENCE,
    ),
    GroupMethodCapability(
        GroupKind.WARPS_WITHIN_BLOCK,
        _ALL_GROUP_QUERY_LEVELS,
        True,
        True,
        GroupFirstReadiness.READY,
        validation_evidence=_GROUP_METHOD_VALIDATION_EVIDENCE,
    ),
    GroupMethodCapability(
        GroupKind.CLUSTER,
        _ALL_GROUP_QUERY_LEVELS,
        True,
        True,
        GroupFirstReadiness.READY,
        validation_evidence=_GROUP_METHOD_VALIDATION_EVIDENCE,
    ),
    GroupMethodCapability(
        GroupKind.GRID,
        _ALL_GROUP_QUERY_LEVELS,
        True,
        True,
        GroupFirstReadiness.READY,
        validation_evidence=_GROUP_METHOD_VALIDATION_EVIDENCE,
    ),
)

GROUP_METHOD_CAPABILITY_BY_KIND = MappingProxyType(
    {capability.group: capability for capability in GROUP_METHOD_CAPABILITIES}
)


def group_method_capability_for(group: GroupKind | str) -> GroupMethodCapability:
    return GROUP_METHOD_CAPABILITY_BY_KIND[GroupKind(group)]


def _selector(**values: str) -> tuple[tuple[str, str], ...]:
    return tuple(values.items())


def _operation(
    name: str,
    family: OperationFamily,
    **selectors: str,
) -> ExportBinding:
    return ExportBinding(
        name,
        ExportBindingKind.OPERATION,
        family,
        selectors=_selector(**selectors),
    )


def _alias(
    name: str,
    family: OperationFamily,
    target_export: str,
    **selectors: str,
) -> ExportBinding:
    return ExportBinding(
        name,
        ExportBindingKind.ALIAS,
        family,
        target_export,
        _selector(**selectors),
    )


def _factory(
    name: str,
    family: OperationFamily,
    target_export: str,
) -> ExportBinding:
    return ExportBinding(
        name,
        ExportBindingKind.FACTORY,
        family,
        target_export,
    )


def _support(name: str) -> ExportBinding:
    return ExportBinding(name, ExportBindingKind.SUPPORT)


BLOCK_OPERATION_BINDINGS = (
    _operation("adjacent_difference", OperationFamily.ADJACENT_DIFFERENCE),
    _operation("discontinuity", OperationFamily.DISCONTINUITY),
    _operation("exchange", OperationFamily.EXCHANGE),
    _operation("histogram", OperationFamily.HISTOGRAM),
    _operation("load", OperationFamily.LOAD),
    _operation("merge_sort_keys", OperationFamily.MERGE_SORT, payload="keys"),
    _operation("merge_sort_pairs", OperationFamily.MERGE_SORT, payload="pairs"),
    _operation("radix_rank", OperationFamily.RADIX_RANK),
    _operation("radix_sort_keys", OperationFamily.RADIX_SORT, payload="keys"),
    _operation("radix_sort_pairs", OperationFamily.RADIX_SORT, payload="pairs"),
    _operation("reduce", OperationFamily.REDUCE),
    _operation("row_sum", OperationFamily.ROW_REDUCE, operator="sum"),
    _operation("run_length_decode", OperationFamily.RUN_LENGTH_DECODE),
    _operation("scan", OperationFamily.SCAN),
    _operation("shuffle", OperationFamily.SHUFFLE),
    _operation("store", OperationFamily.STORE),
    _operation("topk_max_keys", OperationFamily.TOPK, selection="max", payload="keys"),
    _operation(
        "topk_max_pairs", OperationFamily.TOPK, selection="max", payload="pairs"
    ),
    _operation("topk_min_keys", OperationFamily.TOPK, selection="min", payload="keys"),
    _operation(
        "topk_min_pairs", OperationFamily.TOPK, selection="min", payload="pairs"
    ),
)

BLOCK_ALIAS_BINDINGS = (
    _alias(
        "adjacent_difference_subtract_left",
        OperationFamily.ADJACENT_DIFFERENCE,
        "adjacent_difference",
        direction="left",
    ),
    _alias(
        "adjacent_difference_subtract_right",
        OperationFamily.ADJACENT_DIFFERENCE,
        "adjacent_difference",
        direction="right",
    ),
    _alias(
        "discontinuity_flag_heads",
        OperationFamily.DISCONTINUITY,
        "discontinuity",
        mode="heads",
    ),
    _alias(
        "discontinuity_flag_heads_and_tails",
        OperationFamily.DISCONTINUITY,
        "discontinuity",
        mode="heads_and_tails",
    ),
    _alias(
        "discontinuity_flag_tails",
        OperationFamily.DISCONTINUITY,
        "discontinuity",
        mode="tails",
    ),
    *(
        _alias(name, OperationFamily.EXCHANGE, "exchange", mode=mode)
        for name, mode in (
            ("exchange_blocked_to_striped", "blocked_to_striped"),
            ("exchange_blocked_to_warp_striped", "blocked_to_warp_striped"),
            ("exchange_scatter_to_blocked", "scatter_to_blocked"),
            ("exchange_scatter_to_striped", "scatter_to_striped"),
            ("exchange_scatter_to_striped_flagged", "scatter_to_striped_flagged"),
            ("exchange_scatter_to_striped_guarded", "scatter_to_striped_guarded"),
            ("exchange_striped_to_blocked", "striped_to_blocked"),
            ("exchange_warp_striped_to_blocked", "warp_striped_to_blocked"),
        )
    ),
    _alias("exclusive_scan", OperationFamily.SCAN, "scan", scan_mode="exclusive"),
    _alias(
        "exclusive_sum",
        OperationFamily.SCAN,
        "scan",
        scan_mode="exclusive",
        operator="sum",
    ),
    _alias("inclusive_scan", OperationFamily.SCAN, "scan", scan_mode="inclusive"),
    _alias(
        "inclusive_sum",
        OperationFamily.SCAN,
        "scan",
        scan_mode="inclusive",
        operator="sum",
    ),
    _alias(
        "radix_sort_keys_descending",
        OperationFamily.RADIX_SORT,
        "radix_sort_keys",
        order="descending",
        payload="keys",
    ),
    _alias(
        "radix_sort_pairs_descending",
        OperationFamily.RADIX_SORT,
        "radix_sort_pairs",
        order="descending",
        payload="pairs",
    ),
    _alias("shuffle_down", OperationFamily.SHUFFLE, "shuffle", mode="down"),
    _alias("shuffle_offset", OperationFamily.SHUFFLE, "shuffle", mode="offset"),
    _alias("shuffle_rotate", OperationFamily.SHUFFLE, "shuffle", mode="rotate"),
    _alias("shuffle_up", OperationFamily.SHUFFLE, "shuffle", mode="up"),
    _alias("sum", OperationFamily.REDUCE, "reduce", operator="sum"),
)

BLOCK_FACTORY_BINDINGS = tuple(
    _factory(f"make_{target}", family, target)
    for target, family in (
        ("adjacent_difference", OperationFamily.ADJACENT_DIFFERENCE),
        ("discontinuity", OperationFamily.DISCONTINUITY),
        ("exchange", OperationFamily.EXCHANGE),
        ("exclusive_scan", OperationFamily.SCAN),
        ("exclusive_sum", OperationFamily.SCAN),
        ("histogram", OperationFamily.HISTOGRAM),
        ("inclusive_scan", OperationFamily.SCAN),
        ("inclusive_sum", OperationFamily.SCAN),
        ("load", OperationFamily.LOAD),
        ("merge_sort_keys", OperationFamily.MERGE_SORT),
        ("merge_sort_pairs", OperationFamily.MERGE_SORT),
        ("radix_rank", OperationFamily.RADIX_RANK),
        ("radix_sort_keys", OperationFamily.RADIX_SORT),
        ("radix_sort_keys_descending", OperationFamily.RADIX_SORT),
        ("radix_sort_pairs", OperationFamily.RADIX_SORT),
        ("radix_sort_pairs_descending", OperationFamily.RADIX_SORT),
        ("reduce", OperationFamily.REDUCE),
        ("run_length", OperationFamily.RUN_LENGTH_DECODE),
        ("scan", OperationFamily.SCAN),
        ("shuffle", OperationFamily.SHUFFLE),
        ("store", OperationFamily.STORE),
        ("sum", OperationFamily.REDUCE),
        ("topk_max_keys", OperationFamily.TOPK),
        ("topk_max_pairs", OperationFamily.TOPK),
        ("topk_min_keys", OperationFamily.TOPK),
        ("topk_min_pairs", OperationFamily.TOPK),
    )
)

BLOCK_ADAPTER_BINDINGS = (
    ExportBinding(
        "run_length",
        ExportBindingKind.STATEFUL_ADAPTER,
        OperationFamily.RUN_LENGTH_DECODE,
        "run_length_decode",
    ),
)

BLOCK_SUPPORT_BINDINGS = tuple(
    _support(name)
    for name in (
        "BlockAdjacentDifferenceType",
        "BlockDiscontinuityType",
        "BlockExchangeType",
        "BlockRunLengthDecode",
        "BlockShuffleType",
        "TempStorage",
    )
)

BLOCK_EXPORT_BINDINGS = (
    *BLOCK_OPERATION_BINDINGS,
    *BLOCK_ALIAS_BINDINGS,
    *BLOCK_FACTORY_BINDINGS,
    *BLOCK_ADAPTER_BINDINGS,
    *BLOCK_SUPPORT_BINDINGS,
)


WARP_OPERATION_BINDINGS = (
    _operation("exchange", OperationFamily.EXCHANGE),
    _operation("load", OperationFamily.LOAD),
    _operation("merge_sort_keys", OperationFamily.MERGE_SORT, payload="keys"),
    _operation("merge_sort_pairs", OperationFamily.MERGE_SORT, payload="pairs"),
    _operation("reduce", OperationFamily.REDUCE),
    _operation("scan", OperationFamily.SCAN),
    _operation("store", OperationFamily.STORE),
)

WARP_ALIAS_BINDINGS = (
    _alias(
        "exchange_blocked_to_striped",
        OperationFamily.EXCHANGE,
        "exchange",
        mode="blocked_to_striped",
    ),
    _alias(
        "exchange_scatter_to_striped",
        OperationFamily.EXCHANGE,
        "exchange",
        mode="scatter_to_striped",
    ),
    _alias(
        "exchange_striped_to_blocked",
        OperationFamily.EXCHANGE,
        "exchange",
        mode="striped_to_blocked",
    ),
    _alias("exclusive_scan", OperationFamily.SCAN, "scan", scan_mode="exclusive"),
    _alias(
        "exclusive_sum",
        OperationFamily.SCAN,
        "scan",
        scan_mode="exclusive",
        operator="sum",
    ),
    _alias("inclusive_scan", OperationFamily.SCAN, "scan", scan_mode="inclusive"),
    _alias(
        "inclusive_sum",
        OperationFamily.SCAN,
        "scan",
        scan_mode="inclusive",
        operator="sum",
    ),
    _alias("max", OperationFamily.REDUCE, "reduce", operator="max"),
    _alias("min", OperationFamily.REDUCE, "reduce", operator="min"),
    _alias("sum", OperationFamily.REDUCE, "reduce", operator="sum"),
)

WARP_FACTORY_BINDINGS = tuple(
    _factory(f"make_{target}", family, target)
    for target, family in (
        ("exchange", OperationFamily.EXCHANGE),
        ("exclusive_scan", OperationFamily.SCAN),
        ("exclusive_sum", OperationFamily.SCAN),
        ("inclusive_scan", OperationFamily.SCAN),
        ("inclusive_sum", OperationFamily.SCAN),
        ("load", OperationFamily.LOAD),
        ("max", OperationFamily.REDUCE),
        ("merge_sort_keys", OperationFamily.MERGE_SORT),
        ("merge_sort_pairs", OperationFamily.MERGE_SORT),
        ("min", OperationFamily.REDUCE),
        ("reduce", OperationFamily.REDUCE),
        ("store", OperationFamily.STORE),
        ("sum", OperationFamily.REDUCE),
    )
)

WARP_SUPPORT_BINDINGS = tuple(
    _support(name)
    for name in (
        "TempStorage",
        "WarpExchangeType",
    )
)

WARP_EXPORT_BINDINGS = (
    *WARP_OPERATION_BINDINGS,
    *WARP_ALIAS_BINDINGS,
    *WARP_FACTORY_BINDINGS,
    *WARP_SUPPORT_BINDINGS,
)


def _unique_index(items, key):
    index = {}
    for item in items:
        item_key = key(item)
        if item_key in index:
            raise ValueError(f"duplicate registry key {item_key!r}")
        index[item_key] = item
    return MappingProxyType(index)


CAPABILITY_BY_KEY: Mapping[tuple[OperationFamily, GroupKind], Capability] = (
    _unique_index(
        CAPABILITIES, lambda capability: (capability.family, capability.group)
    )
)
BLOCK_BINDING_BY_NAME: Mapping[str, ExportBinding] = _unique_index(
    BLOCK_EXPORT_BINDINGS, lambda binding: binding.name
)
WARP_BINDING_BY_NAME: Mapping[str, ExportBinding] = _unique_index(
    WARP_EXPORT_BINDINGS, lambda binding: binding.name
)


def capability_for(
    family: OperationFamily | str,
    group: GroupKind | str,
) -> Capability:
    """Return the capability for one canonical family and group kind."""

    return CAPABILITY_BY_KEY[OperationFamily(family), GroupKind(group)]


def binding_for(group: GroupKind | str, name: str) -> ExportBinding:
    """Return the scoped public-export binding for ``name``."""

    group = GroupKind(group)
    if group not in {GroupKind.BLOCK, GroupKind.WARP}:
        raise KeyError(f"{group.value!r} has no scoped export bindings")
    bindings = (
        BLOCK_BINDING_BY_NAME if group is GroupKind.BLOCK else WARP_BINDING_BY_NAME
    )
    return bindings[name]


def resolved_binding_selectors(
    group: GroupKind | str,
    name: str,
) -> Mapping[str, str]:
    """Resolve selectors through an alias, factory, or adapter target chain."""

    group = GroupKind(group)
    binding = binding_for(group, name)
    bindings = (
        BLOCK_BINDING_BY_NAME if group is GroupKind.BLOCK else WARP_BINDING_BY_NAME
    )
    selectors: dict[str, str] = {}
    visited: set[str] = set()
    while True:
        if binding.name in visited:
            raise ValueError(f"cyclic export binding at {binding.name!r}")
        visited.add(binding.name)
        for selector_name, selector_value in binding.selectors:
            previous = selectors.setdefault(selector_name, selector_value)
            if previous != selector_value:
                raise ValueError(
                    f"conflicting {selector_name!r} selectors in binding chain"
                )
        if binding.target_export is None:
            break
        binding = bindings[binding.target_export]
    return MappingProxyType(selectors)


def group_first_planner_models_binding(group: GroupKind | str, name: str) -> bool:
    """Whether the shared planner models an exact scoped export binding.

    Canonical operations without fixed selectors return whether any variant of
    the family is modeled. Selector-bearing aliases must also satisfy every
    selector restriction recorded by the capability. This does not imply that
    the provider is converted, the root API is exposed, or runtime gates pass.
    """

    group = GroupKind(group)
    binding = binding_for(group, name)
    if binding.family is None:
        return False
    capability = capability_for(binding.family, group)
    if not capability.group_first_operand_forms:
        return False
    selected = resolved_binding_selectors(group, name)
    return all(
        selector.name not in selected
        or selected[selector.name] in selector.accepted_values
        for selector in capability.group_first_selector_support
    )


__all__ = [
    "AlternateLoweringRoute",
    "AlternateLoweringRouteCase",
    "ApiAvailability",
    "ApiStability",
    "BLOCK_ADAPTER_BINDINGS",
    "BLOCK_ALIAS_BINDINGS",
    "BLOCK_BINDING_BY_NAME",
    "BLOCK_EXPORT_BINDINGS",
    "BLOCK_FACTORY_BINDINGS",
    "BLOCK_OPERATION_BINDINGS",
    "BLOCK_SUPPORT_BINDINGS",
    "CAPABILITIES",
    "CAPABILITY_BY_KEY",
    "Capability",
    "CppApi",
    "ExportBinding",
    "ExportBindingKind",
    "GroupFirstReadiness",
    "GroupFirstStage",
    "GroupKind",
    "GroupMethodCapability",
    "GROUP_METHOD_CAPABILITIES",
    "GROUP_METHOD_CAPABILITY_BY_KIND",
    "OperandForm",
    "OperandKind",
    "OperationFamily",
    "PayloadAdapterProvenance",
    "ProvenanceKind",
    "ProviderProvenance",
    "SelectorSupport",
    "WARP_ALIAS_BINDINGS",
    "WARP_BINDING_BY_NAME",
    "WARP_EXPORT_BINDINGS",
    "WARP_FACTORY_BINDINGS",
    "WARP_OPERATION_BINDINGS",
    "WARP_SUPPORT_BINDINGS",
    "binding_for",
    "capability_for",
    "group_first_planner_models_binding",
    "group_method_capability_for",
    "resolved_binding_selectors",
]
