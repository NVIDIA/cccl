# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe which threads participate in a cooperative operation.

``ThreadHierarchy`` holds the launch dimensions known to a planner.
``ThreadGroup`` selects a physical group or partitions it into smaller groups.
Public ``this_*`` factories start with the current launch; a backend later
resolves the dimensions from ``LaunchFacts``.

A descriptor is a Python planning value. Constructing one does not launch
a kernel, synchronize threads, or establish that a primitive supports
the requested group. Group resolution and operation planning check those
requirements before a backend uses the CUDAX declaration helpers here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from types import GenericAlias
from typing import Any, TypeVar

# Hierarchy levels are the coordinate spaces accepted by rank and count queries.
THREAD_LEVELS = frozenset({"thread", "warp", "block", "cluster", "grid"})
# Physical groups use CUDA's thread, warp, block, cluster, and grid levels.
PHYSICAL_GROUP_KINDS = frozenset({"thread", "warp", "block", "cluster", "grid"})
MAPPED_GROUP_KINDS = frozenset({"threads_within_warp", "warps_within_block"})
THREAD_GROUP_KINDS = PHYSICAL_GROUP_KINDS | MAPPED_GROUP_KINDS
COMPLETE_WARP_GROUP_KINDS = frozenset({"warp"}) | MAPPED_GROUP_KINDS
THREAD_GROUP_QUERY_DTYPE_NAMES = frozenset(
    {"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}
)
_ThreadGroupT = TypeVar("_ThreadGroupT", bound="ThreadGroup")
_CPP_LEVEL_EXPR = {
    "thread": "::cuda::gpu_thread",
    "warp": "::cuda::warp",
    "block": "::cuda::block",
    "cluster": "::cuda::cluster",
    "grid": "::cuda::grid",
}


class CoopCompilerContextRequiredError(RuntimeError):
    """Report a cooperative value used outside the compiler stage it needs."""


def _compiler_method_marker(method: str) -> Any:
    raise CoopCompilerContextRequiredError(
        f"cuda.coop.ThreadGroup.{method} requires compiler-owned activation "
        "or a qualified backend import before compilation"
    )


def normalize_thread_dim(
    value: Any,
    *,
    scope: str,
    label: str,
) -> tuple[int, int, int]:
    """Convert a positive launch shape to an ``(x, y, z)`` tuple.

    Accept an integer or a tuple/list of one to three integers. Pad missing
    axes with one so equivalent spellings share the same descriptor and
    cache identity. Booleans, empty shapes, and nonpositive dimensions are
    invalid. ``scope`` and ``label`` identify the caller in diagnostics.
    """

    if isinstance(value, bool):
        raise TypeError(f"{scope} {label} shape must be int-like")
    if isinstance(value, int):
        dims = (value,)
    elif isinstance(value, (tuple, list)):
        dims = tuple(value)
        if not dims:
            raise ValueError(f"{scope} {label} shape cannot be empty")
        if len(dims) > 3:
            raise ValueError(f"{scope} {label} shape must have at most 3 dimensions")
    else:
        raise TypeError(f"{scope} {label} shape must be an int or tuple/list")

    normalized = []
    for dim in dims:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError(f"{scope} {label} dimensions must be integers")
        if dim <= 0:
            raise ValueError(f"{scope} {label} dimensions must be positive")
        normalized.append(dim)

    while len(normalized) < 3:
        normalized.append(1)
    return tuple(normalized)  # type: ignore[return-value]


def normalize_thread_level(level: str, *, scope: str, feature: str) -> str:
    """Validate a hierarchy level and accept ``gpu_thread`` as ``thread``.

    Use ``scope`` and ``feature`` to identify an invalid level in the error.
    """

    if level == "gpu_thread":
        level = "thread"
    if level not in THREAD_LEVELS:
        names = ", ".join(sorted(THREAD_LEVELS))
        raise ValueError(f"{scope}.{feature} level must be one of: {names}")
    return level


def normalize_thread_group_kind(kind: str, *, scope: str, feature: str) -> str:
    """Validate a physical or mapped group kind.

    Accept ``gpu_thread`` as ``thread``. Group kinds include static partitions
    as well as physical levels; ``scope`` and ``feature`` identify invalid
    input in the diagnostic.
    """

    if kind == "gpu_thread":
        kind = "thread"
    if kind not in THREAD_GROUP_KINDS:
        names = ", ".join(sorted(THREAD_GROUP_KINDS))
        raise ValueError(f"{scope}.{feature} group kind must be one of: {names}")
    return kind


def validate_thread_group_query_dtype(dtype: Any, *, scope: str) -> str:
    """Validate the integral result domain shared by thread-group backends."""

    token = getattr(dtype, "name", None)
    if token is None:
        token = getattr(dtype, "__name__", None)
    if token is None:
        token = str(dtype)
    token = str(token).lower()
    if token not in THREAD_GROUP_QUERY_DTYPE_NAMES:
        names = ", ".join(sorted(THREAD_GROUP_QUERY_DTYPE_NAMES))
        raise TypeError(f"{scope}.ThreadGroup query dtype must be one of: {names}")
    return token


def _thread_count(block_dim: tuple[int, int, int] | None) -> int | None:
    if block_dim is None:
        return None
    return reduce(mul, block_dim, 1)


def _dims_token(prefix: str, dims: tuple[int, int, int]) -> str:
    x, y, z = dims
    if y == 1 and z == 1:
        return f"{prefix}{x}"
    if z == 1:
        return f"{prefix}{x}x{y}"
    return f"{prefix}{x}x{y}x{z}"


@dataclass(frozen=True, init=False)
class ThreadHierarchy:
    """Describe the launch dimensions known during group planning.

    Public construction describes the current kernel launch with unresolved
    dimensions. Backends supply exact extents from ``LaunchFacts`` when they
    resolve a group. Callers cannot pass independent launch dimensions to
    the public constructor.

    Attributes
    ----------
    block_dim : tuple of int or None
        Threads along each axis of one block, once known.
    cluster_dim : tuple of int or None
        Blocks along each axis of one cluster, once needed and known.
    grid_dim : tuple of int or None
        Clusters along each axis of the resolved grid. On a non-cluster
        launch, each cluster contains one block.
    implicit : bool
        Whether this descriptor still refers to the current launch without
        explicit dimensions. A resolved hierarchy can leave higher levels
        unknown if the selected group does not need them.
    """

    block_dim: tuple[int, int, int] | None
    grid_dim: tuple[int, int, int] | None
    cluster_dim: tuple[int, int, int] | None
    implicit: bool

    def __init__(self) -> None:
        object.__setattr__(self, "block_dim", None)
        object.__setattr__(self, "grid_dim", None)
        object.__setattr__(self, "cluster_dim", None)
        object.__setattr__(self, "implicit", True)

    @classmethod
    def _resolved(
        cls,
        *,
        block_dim: int | tuple[int, ...] | list[int],
        grid_dim: int | tuple[int, ...] | list[int] | None = None,
        cluster_dim: int | tuple[int, ...] | list[int] | None = None,
    ) -> ThreadHierarchy:
        """Construct a hierarchy from dimensions supplied by a planner.

        Normalize each supplied shape and mark the result as explicit. The
        caller must obtain and check the launch facts; this constructor cannot
        verify an actual launch. ``grid_dim`` counts clusters, so the resolver
        must first convert physical grid dimensions when needed.
        """

        hierarchy = object.__new__(cls)
        object.__setattr__(
            hierarchy,
            "block_dim",
            normalize_thread_dim(
                block_dim,
                scope="ThreadHierarchy",
                label="block",
            ),
        )
        object.__setattr__(
            hierarchy,
            "grid_dim",
            None
            if grid_dim is None
            else normalize_thread_dim(
                grid_dim,
                scope="ThreadHierarchy",
                label="grid",
            ),
        )
        object.__setattr__(
            hierarchy,
            "cluster_dim",
            None
            if cluster_dim is None
            else normalize_thread_dim(
                cluster_dim,
                scope="ThreadHierarchy",
                label="cluster",
            ),
        )
        object.__setattr__(hierarchy, "implicit", False)
        return hierarchy

    @classmethod
    def current(cls) -> ThreadHierarchy:
        """Describe the current launch before its dimensions are known."""

        return cls()

    @property
    def is_static(self) -> bool:
        """Return whether a planner supplied explicit hierarchy dimensions.

        Higher levels can still be unknown. Use ``has_static_extents_for``
        when a particular group level needs them.
        """

        return not self.implicit

    @property
    def block_thread_count(self) -> int | None:
        """Return the block thread count, or ``None`` if it is unknown."""

        return _thread_count(self.block_dim)  # type: ignore[arg-type]

    @property
    def symbol_suffix(self) -> str:
        """Encode known hierarchy dimensions for generated symbol names.

        Use ``current`` for an unresolved launch and include each known outer
        level before the block shape.
        """

        if self.implicit:
            return "current"
        parts: list[str] = []
        if self.grid_dim is not None:
            parts.append(_dims_token("g", self.grid_dim))  # type: ignore[arg-type]
        if self.cluster_dim is not None:
            parts.append(_dims_token("c", self.cluster_dim))  # type: ignore[arg-type]
        parts.append(self.block_dim_token)
        return "_".join(parts)

    @property
    def block_dim_token(self) -> str:
        """Encode the block shape, or use ``current`` if it is unknown."""

        if self.block_dim is None:
            return "current"
        return _dims_token("b", self.block_dim)

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """Identify the dimensions and their current-launch status."""

        return self.block_dim, self.grid_dim, self.cluster_dim, self.implicit

    def has_static_extents_for(self, group_kind: str) -> bool:
        """Check whether this hierarchy has the extents used by a group level.

        Thread groups have a known size even in an implicit hierarchy. Other
        physical groups need the block shape and, for cluster or grid groups,
        the corresponding outer shape. This checks available dimensions;
        operation support and launch capabilities are separate checks.
        """

        group_kind = normalize_thread_level(
            group_kind,
            scope="ThreadHierarchy",
            feature="has_static_extents_for",
        )
        if self.implicit:
            return group_kind == "thread"
        if group_kind in {"thread", "warp", "block"}:
            return self.block_dim is not None
        if group_kind == "cluster":
            return self.block_dim is not None and self.cluster_dim is not None
        if group_kind == "grid":
            return self.block_dim is not None and self.grid_dim is not None
        return False


Hierarchy = ThreadHierarchy


@dataclass(frozen=True)
class GroupByMapping:
    """Describe a static partition of a physical warp or block.

    ``ThreadGroup.group_by`` creates this record. A warp partitions into
    sets of threads; a block partitions into sets of complete physical
    warps. The record keeps the unit count and synchronization choice so
    C++ generation and planning agree on each subgroup's membership.

    Parameters
    ----------
    unit : str
        Unit placed in each subgroup: ``thread`` or ``warp``.
    parent : str
        Physical parent level: ``warp`` or ``block``. ``ThreadGroup`` checks
        that it matches the selected unit and mapped group kind.
    count : int
        Positive number of units in one subgroup, known during compilation.
    exhaustive : bool
        Whether the count must divide the parent's unit count exactly.
        A non-exhaustive mapping can leave units outside complete groups.
    synchronizer : str
        Must be ``lane`` for thread units and ``barrier`` for warp units.
        Construction rejects any other pairing. Code generation emits the
        matching synchronizer.
    """

    unit: str
    parent: str
    count: int
    exhaustive: bool
    synchronizer: str

    def __post_init__(self) -> None:
        """Check partition controls before attaching them to a group."""

        if self.unit not in {"thread", "warp"}:
            raise ValueError("GroupByMapping unit must be thread or warp")
        if self.parent not in {"warp", "block"}:
            raise ValueError("GroupByMapping parent must be warp or block")
        if not isinstance(self.count, int) or isinstance(self.count, bool):
            raise TypeError("GroupByMapping count must be a static integer")
        if self.count <= 0:
            raise ValueError("GroupByMapping count must be positive")
        if not isinstance(self.exhaustive, bool):
            raise TypeError("GroupByMapping exhaustive must be a bool")
        expected_synchronizer = "lane" if self.unit == "thread" else "barrier"
        if self.synchronizer != expected_synchronizer:
            raise ValueError(
                f"GroupByMapping {self.unit} unit requires "
                f"{expected_synchronizer!r} synchronization"
            )

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """Identify partition size, coverage, and synchronization."""

        return (
            self.unit,
            self.parent,
            self.count,
            self.exhaustive,
            self.synchronizer,
        )


def _validate_mapped_group_extent(
    kind: str,
    hierarchy: ThreadHierarchy,
    mapping: GroupByMapping,
) -> None:
    """Check a partition against the parent dimensions already known.

    Mapped groups require complete physical warps. Their count cannot exceed
    the parent's units, and an exhaustive count must divide those units.
    For a block with an unknown size, defer these extent checks to launch
    resolution. A warp's 32-thread extent is already known.
    """

    block_threads = hierarchy.block_thread_count
    if block_threads is not None and block_threads % 32 != 0:
        raise ValueError(
            "mapped group_by requires an enclosing block composed of complete warps"
        )
    if kind == "threads_within_warp":
        parent_units = 32
        if mapping.count > parent_units:
            raise ValueError("warp group_by count cannot exceed 32 threads")
    else:
        if block_threads is None:
            return
        parent_units = block_threads // 32
        if mapping.count > parent_units:
            raise ValueError("block group_by count cannot exceed the parent warp count")
    if mapping.exhaustive and parent_units % mapping.count != 0:
        raise ValueError(
            "exhaustive ThreadGroup.group_by requires the count to divide "
            "the parent unit count"
        )


@dataclass(frozen=True)
class ThreadGroup:
    """Describe the participants in a cooperative operation.

    Use the ``this_*`` factories to describe groups in the current kernel
    launch. Constructing a descriptor does not synchronize threads or launch
    a kernel. Load and Store support blocks, physical warps, and logical
    warps; see :ref:`thread groups <coop-thread-groups>`.

    Group descriptors can also be constructed in ordinary Python. Runtime
    rank, size, membership, and synchronization queries require a
    supported kernel compiler.
    """

    __class_getitem__ = classmethod(GenericAlias)

    kind: str
    hierarchy: ThreadHierarchy = field(default_factory=ThreadHierarchy.current)
    parent: ThreadGroup | None = None
    mapping: GroupByMapping | None = None
    # Source labels do not affect structural identity or cache keys.
    # Planners can still use them to preserve public API policy.
    source: str = field(default="explicit", compare=False, hash=False)

    def __post_init__(self) -> None:
        """Check that the kind, parent, and partition describe one group.

        Mapped groups must use the expected physical parent and the same
        hierarchy. Physical groups carry no mapping metadata. Known dimensions
        also let construction reject impossible partition sizes early.
        """

        kind = normalize_thread_group_kind(
            self.kind,
            scope="ThreadGroup",
            feature="kind",
        )
        object.__setattr__(self, "kind", kind)

        hierarchy = self.hierarchy
        if not isinstance(hierarchy, ThreadHierarchy):
            raise TypeError("ThreadGroup hierarchy must be a ThreadHierarchy")

        if kind in MAPPED_GROUP_KINDS:
            if not isinstance(self.parent, ThreadGroup):
                raise TypeError("mapped ThreadGroup requires a parent ThreadGroup")
            if not isinstance(self.mapping, GroupByMapping):
                raise TypeError("mapped ThreadGroup requires GroupByMapping")
            expected_parent = "warp" if kind == "threads_within_warp" else "block"
            expected_unit = "thread" if kind == "threads_within_warp" else "warp"
            if self.parent.kind != expected_parent:
                raise ValueError(f"{kind} requires a physical {expected_parent} parent")
            if (
                self.mapping.parent != expected_parent
                or self.mapping.unit != expected_unit
            ):
                raise ValueError(f"{kind} mapping does not match its group kind")
            if self.parent.hierarchy != hierarchy:
                raise ValueError("mapped ThreadGroup hierarchy must match its parent")
            _validate_mapped_group_extent(kind, hierarchy, self.mapping)
        elif self.parent is not None or self.mapping is not None:
            raise ValueError("physical ThreadGroup cannot carry mapping metadata")

    @property
    def block_dim(self) -> tuple[int, int, int] | None:
        """Return the resolved block dimensions, or ``None`` if unknown."""

        return self.hierarchy.block_dim

    @property
    def group_thread_count(self) -> int:
        """Return the known number of threads, or reject an unresolved size.

        A physical warp always has an extent of 32. The operation planner
        still checks whether all 32 lanes can participate. The descriptor does
        not observe active lanes. Use ``static_size`` to return ``None``
        instead of raising ``ValueError`` when the size is unknown.
        """

        count = self.static_size
        if count is None:
            raise ValueError(
                f"ThreadGroup.{self.kind} uses a runtime hierarchy with no "
                "static group size"
            )
        return count

    @property
    def static_size(self) -> int | None:
        """Return the known group thread count, or ``None`` if unresolved.

        Thread, warp, and mapped group sizes follow from their kind and
        mapping. Block, cluster, and grid sizes need hierarchy dimensions. The
        count describes membership, but does not prove that all threads in
        the parent participate.
        """

        if self.kind == "thread":
            return 1
        if self.kind == "warp":
            return 32
        if self.kind == "threads_within_warp":
            assert self.mapping is not None
            return self.mapping.count
        if self.kind == "warps_within_block":
            assert self.mapping is not None
            return self.mapping.count * 32
        hierarchy = self.hierarchy
        assert hierarchy is not None
        if self.kind == "block":
            return hierarchy.block_thread_count
        if self.kind == "cluster":
            block_threads = hierarchy.block_thread_count
            if block_threads is None or hierarchy.cluster_dim is None:
                return None
            cluster_blocks = _thread_count(hierarchy.cluster_dim)
            assert cluster_blocks is not None
            return block_threads * cluster_blocks
        if self.kind == "grid":
            block_threads = hierarchy.block_thread_count
            if block_threads is None or hierarchy.grid_dim is None:
                return None
            grid_groups = _thread_count(hierarchy.grid_dim)
            cluster_blocks = (
                1
                if hierarchy.cluster_dim is None
                else _thread_count(hierarchy.cluster_dim)
            )
            assert grid_groups is not None
            assert cluster_blocks is not None
            return block_threads * cluster_blocks * grid_groups
        return None

    @property
    def is_current(self) -> bool:
        """Return whether the hierarchy still needs launch resolution."""

        return self.hierarchy.implicit  # type: ignore[union-attr]

    @property
    def is_static(self) -> bool:
        """Return whether the hierarchy has the extents this group needs.

        Mapped groups use their physical parent's result. A known subgroup
        size alone does not supply an unknown enclosing block shape.
        """

        if self.kind in MAPPED_GROUP_KINDS:
            assert self.parent is not None
            return self.parent.is_static
        return self.hierarchy.has_static_extents_for(self.kind)  # type: ignore[union-attr]

    @property
    def parent_unit_count(self) -> int | None:
        """Return the parent's known unit count for a mapped group.

        A warp parent has 32 thread units. A block parent has one unit per
        complete physical warp. Return ``None`` for a physical group or an
        unknown parent count.
        """

        if self.kind == "threads_within_warp":
            return 32
        if self.kind == "warps_within_block":
            hierarchy = self.hierarchy
            assert hierarchy is not None
            block_threads = hierarchy.block_thread_count
            if block_threads is None or block_threads % 32 != 0:
                return None
            return block_threads // 32
        return None

    @property
    def groups_per_parent(self) -> int | None:
        """Count complete mapped subgroups within one physical parent.

        Exclude any remainder allowed by a non-exhaustive mapping. Return
        ``None`` when the group is physical or its parent count is unknown.
        """

        if self.mapping is None:
            return None
        parent_units = self.parent_unit_count
        if parent_units is None:
            return None
        return parent_units // self.mapping.count

    @property
    def remainder_count(self) -> int | None:
        """Count parent units left outside complete mapped subgroups.

        The unit is a thread for a warp parent and a warp for a block parent.
        Return ``None`` for a physical group or an unknown parent count.
        """

        if self.mapping is None:
            return None
        parent_units = self.parent_unit_count
        if parent_units is None:
            return None
        return parent_units % self.mapping.count

    @property
    def complete_membership(self) -> bool | None:
        """Report whether the mapping covers all units of its physical parent.

        Return ``None`` until a mapped parent's unit count is known. Physical
        groups return true because no mapping excludes members. Execution can
        still require complete warps and converged participation.
        """

        if self.mapping is None:
            return True
        remainder = self.remainder_count
        if remainder is None:
            return None
        return remainder == 0

    @property
    def symbol_suffix(self) -> str:
        """Encode the group kind, partition, and hierarchy for C++ symbols."""

        if self.mapping is None:
            return f"{self.kind}_{self.hierarchy.symbol_suffix}"  # type: ignore[union-attr]
        mode = "all" if self.mapping.exhaustive else "partial"
        return (
            f"{self.kind}_{self.mapping.count}_{mode}_"
            f"{self.hierarchy.symbol_suffix}"  # type: ignore[union-attr]
        )

    @property
    def block_dim_token(self) -> str:
        return self.hierarchy.block_dim_token  # type: ignore[union-attr]

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """Identify group shape and mapping without their source label.

        Include the parent for mapped groups. Planners can use ``source`` for
        API policy, but that label does not change this structural identity.
        """

        if self.mapping is None:
            return self.kind, self.hierarchy.semantic_key  # type: ignore[union-attr]
        assert self.parent is not None
        return (
            self.kind,
            self.parent.semantic_key,
            self.mapping.semantic_key,
            self.hierarchy.semantic_key,  # type: ignore[union-attr]
        )

    # Keep runtime annotations dependency-free on Python 3.10.
    def with_hierarchy(  # noqa: PYI019
        self: _ThreadGroupT,
        hierarchy: ThreadHierarchy,
        *,
        source: str = "resolved",
    ) -> _ThreadGroupT:
        """Copy the descriptor with a given hierarchy and source label.

        Keep its concrete backend type. For a mapped group, give its physical
        parent the same hierarchy so both descriptions remain consistent.
        Construction rechecks the mapping against the known dimensions.
        """

        if self.mapping is None:
            return type(self)(kind=self.kind, hierarchy=hierarchy, source=source)
        assert self.parent is not None
        return type(self)(
            kind=self.kind,
            hierarchy=hierarchy,
            parent=self.parent.with_hierarchy(hierarchy, source=source),
            mapping=self.mapping,
            source=source,
        )

    # Keep runtime annotations dependency-free on Python 3.10.
    def group_by(  # noqa: PYI019
        self: _ThreadGroupT,
        count: int,
        *,
        exhaustive: bool = True,
    ) -> _ThreadGroupT:
        """Partition a physical warp by threads or a block by warps.

        Parameters
        ----------
        count : int
            Positive compile-time number of units in each subgroup. For a warp
            parent, the unit is one thread; for a block parent, it is one
            physical warp. Thus ``this_warp().group_by(8)`` describes eight
            lanes, while ``this_block().group_by(2)`` describes 64 threads.
        exhaustive : bool, optional
            Compile-time flag, default ``True``. An exhaustive partition must
            divide the parent's unit count exactly. ``False`` permits a
            remainder outside the complete groups. Each primitive still
            determines which partitions it supports.

        Returns
        -------
        cuda.coop.ThreadGroup
            A descriptor for the subgroup containing the calling thread.
            Nested partitions are unsupported. Load and Store support logical
            warp widths of 1, 2, 4, 8, 16, or 32; mapped groups of physical
            warps are not Load or Store targets.

        Examples
        --------
        Descriptors can be inspected without compiling or launching a kernel:

        .. code-block:: python

            from cuda import coop

            group = coop.this_warp().group_by(8)
            assert group.kind == "threads_within_warp"
            assert group.static_size == 8
        """

        if self.mapping is not None:
            raise NotImplementedError("nested ThreadGroup.group_by is not supported")
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError("ThreadGroup.group_by count must be a static integer")
        if count <= 0:
            raise ValueError("ThreadGroup.group_by count must be positive")
        if not isinstance(exhaustive, bool):
            raise TypeError("ThreadGroup.group_by exhaustive must be a bool")

        if self.kind == "warp":
            kind = "threads_within_warp"
            unit = "thread"
            synchronizer = "lane"
        elif self.kind == "block":
            kind = "warps_within_block"
            unit = "warp"
            synchronizer = "barrier"
        else:
            raise NotImplementedError(
                "ThreadGroup.group_by supports only physical warp and block parents"
            )

        mapping = GroupByMapping(
            unit=unit,
            parent=self.kind,
            count=count,
            exhaustive=exhaustive,
            synchronizer=synchronizer,
        )
        return type(self)(
            kind=kind,
            hierarchy=self.hierarchy,
            parent=self,
            mapping=mapping,
            source="group_by",
        )

    def rank(self, level: str = "thread") -> Any:
        """Return this group's rank relative to another hierarchy level."""

        del level
        return _compiler_method_marker("rank")

    def count(self, level: str = "thread") -> Any:
        """Return this group's count relative to another hierarchy level."""

        del level
        return _compiler_method_marker("count")

    def rank_as(self, dtype: Any = None, level: str = "thread") -> Any:
        """Return the group rank converted to an integral dtype."""

        del dtype, level
        return _compiler_method_marker("rank_as")

    def count_as(self, dtype: Any = None, level: str = "thread") -> Any:
        """Return the group count converted to an integral dtype."""

        del dtype, level
        return _compiler_method_marker("count_as")

    def sync(self) -> None:
        """Synchronize the participating members of this group."""

        _compiler_method_marker("sync")

    def sync_aligned(self) -> None:
        """Synchronize an aligned group in converged control flow."""

        _compiler_method_marker("sync_aligned")

    def is_member(self) -> Any:
        """Return whether the current thread belongs to this group."""

        return _compiler_method_marker("is_member")


def make_thread_group(
    kind: str,
    *,
    group_type: type[_ThreadGroupT] = ThreadGroup,
    scope: str = "cuda.coop",
) -> _ThreadGroupT:
    """Create a physical group descriptor for the current launch.

    Backends can supply a ``group_type`` subclass while using the same kind
    normalization and unresolved hierarchy. ``scope`` names the API in an
    invalid-kind diagnostic.
    """

    kind = normalize_thread_level(kind, scope=scope, feature="ThreadGroup")
    return group_type(
        kind=kind,
        hierarchy=ThreadHierarchy.current(),
        source="current",
    )


def _cpp_dims_expr(level: str, dims: tuple[int, int, int]) -> str:
    x, y, z = dims
    if y == 1 and z == 1:
        return f"::cuda::{level}_dims<{x}>()"
    if z == 1:
        return f"::cuda::{level}_dims<{x}, {y}>()"
    return f"::cuda::{level}_dims<{x}, {y}, {z}>()"


def render_hierarchy_decl(
    hierarchy: ThreadHierarchy,
    *,
    var_name: str = "hierarchy",
    indent: str = "  ",
) -> list[str]:
    """Generate a CUDAX hierarchy declaration from known dimensions.

    An implicit hierarchy needs no declaration and returns an empty list.
    For an explicit hierarchy, emit the known grid and cluster levels before
    the required block level. ``var_name`` and ``indent`` control the
    surrounding generated source.
    """

    if hierarchy.implicit:
        return []
    exprs: list[str] = []
    if hierarchy.grid_dim is not None:
        exprs.append(_cpp_dims_expr("grid", hierarchy.grid_dim))
    if hierarchy.cluster_dim is not None:
        exprs.append(_cpp_dims_expr("cluster", hierarchy.cluster_dim))
    assert hierarchy.block_dim is not None
    exprs.append(_cpp_dims_expr("block", hierarchy.block_dim))
    if len(exprs) == 1:
        return [f"{indent}auto {var_name} = ::cuda::hierarchy{{{exprs[0]}}};"]
    lines = [f"{indent}auto {var_name} = ::cuda::hierarchy{{"]
    for idx, expr in enumerate(exprs):
        comma = "," if idx < len(exprs) - 1 else ""
        lines.append(f"{indent}    {expr}{comma}")
    lines.append(f"{indent}}};")
    return lines


def render_group_decl(
    group: ThreadGroup,
    *,
    var_name: str = "group",
    hierarchy_var: str = "hierarchy",
    indent: str = "  ",
) -> str:
    """Generate one CUDAX declaration for a physical group.

    Use the supplied ``hierarchy_var`` for an explicit hierarchy, or CUDAX's
    implicit hierarchy for the current launch. The caller must declare an
    explicit hierarchy first. Mapped groups need additional declarations;
    reject them here and use ``render_group_decl_lines``.
    """

    if group.mapping is not None:
        raise ValueError(
            "render_group_decl supports physical groups only; use "
            "render_group_decl_lines for mapped groups"
        )
    assert group.hierarchy is not None
    if group.hierarchy.implicit:
        return (
            f"{indent}::cuda::experimental::coop::this_{group.kind} "
            f"{var_name}{{::cuda::experimental::implicit_hierarchy()}};"
        )
    return (
        f"{indent}::cuda::experimental::coop::this_{group.kind} "
        f"{var_name}{{{hierarchy_var}}};"
    )


def render_group_decl_lines(
    group: ThreadGroup,
    *,
    var_name: str = "group",
    hierarchy_var: str = "hierarchy",
    indent: str = "  ",
) -> list[str]:
    """Generate a CUDAX group and any declarations its mapping needs.

    Physical groups need one declaration. A thread subgroup also needs its
    physical warp parent and a lane synchronizer. A subgroup of physical
    warps needs its block parent and shared barrier storage, with one
    barrier slot per complete subgroup.

    The mapped block's subgroup count must be known to size that storage;
    otherwise raise ``ValueError``. These lines describe the group and its
    storage. They do not establish the enclosing launch's capabilities or
    prove that a primitive supports the group.
    """

    if group.mapping is None:
        return [
            render_group_decl(
                group,
                var_name=var_name,
                hierarchy_var=hierarchy_var,
                indent=indent,
            )
        ]

    assert group.parent is not None
    mapping = group.mapping
    exhaustive = "true" if mapping.exhaustive else "false"
    mapping_args = str(mapping.count)
    if not mapping.exhaustive:
        mapping_args = (
            f"::cuda::experimental::coop::non_exhaustive, {mapping.count}"
        )
    parent_name = f"{var_name}_parent"
    lines = [
        render_group_decl(
            group.parent,
            var_name=parent_name,
            hierarchy_var=hierarchy_var,
            indent=indent,
        )
    ]
    if group.kind == "threads_within_warp":
        lines.extend(
            [
                f"{indent}::cuda::experimental::coop::generic_group {var_name}{{",
                f"{indent}    ::cuda::gpu_thread, {parent_name},",
                (
                    f"{indent}    ::cuda::experimental::coop::group_by<"
                    f"{mapping.count}, {exhaustive}>{{{mapping_args}}},"
                ),
                f"{indent}    ::cuda::experimental::coop::lane_synchronizer{{}}}};",
            ]
        )
        return lines

    groups_per_parent = group.groups_per_parent
    if groups_per_parent is None:
        raise ValueError("mapped warp group requires a static parent group count")
    lines.extend(
        [
            f"{indent}using {var_name}_barriers_type =",
            (
                f"{indent}    ::cuda::barrier<::cuda::thread_scope_block>"
                f"[{groups_per_parent}];"
            ),
            f"{indent}__shared__ ::cuda::std::aligned_storage_t<",
            f"{indent}    sizeof({var_name}_barriers_type),",
            (
                f"{indent}    alignof({var_name}_barriers_type)> "
                f"{var_name}_barriers_storage;"
            ),
            f"{indent}auto& {var_name}_barriers =",
            f"{indent}    reinterpret_cast<{var_name}_barriers_type&>(",
            f"{indent}        {var_name}_barriers_storage);",
            f"{indent}::cuda::experimental::coop::generic_group {var_name}{{",
            f"{indent}    ::cuda::warp, {parent_name},",
            (
                f"{indent}    ::cuda::experimental::coop::group_by<"
                f"{mapping.count}, {exhaustive}>{{{mapping_args}}},"
            ),
            (
                f"{indent}    ::cuda::experimental::coop::barrier_synchronizer{{"
                f"{var_name}_barriers}}}};"
            ),
        ]
    )
    return lines


def cpp_level_expr(level: str) -> str:
    """Translate a hierarchy level to its CUDAX tag expression.

    For example, ``thread`` selects ``::cuda::gpu_thread``. Reuse level
    normalization so the accepted Python spellings stay consistent.
    """

    return _CPP_LEVEL_EXPR[
        normalize_thread_level(level, scope="cuda.coop", feature="cpp_level_expr")
    ]


def this_thread() -> ThreadGroup:
    """Describe the calling thread as a one-thread group."""

    return make_thread_group("thread")


def this_warp() -> ThreadGroup:
    """Describe the calling thread's physical 32-thread warp."""

    return make_thread_group("warp")


def this_block() -> ThreadGroup:
    """Describe the calling thread's block in the current launch."""

    return make_thread_group("block")


def this_cluster() -> ThreadGroup:
    """Describe the calling thread's cluster for later launch validation."""

    return make_thread_group("cluster")


def this_grid() -> ThreadGroup:
    """Describe the launch grid for later cooperative-launch validation."""

    return make_thread_group("grid")


__all__ = [
    "COMPLETE_WARP_GROUP_KINDS",
    "MAPPED_GROUP_KINDS",
    "PHYSICAL_GROUP_KINDS",
    "THREAD_GROUP_KINDS",
    "THREAD_GROUP_QUERY_DTYPE_NAMES",
    "THREAD_LEVELS",
    "CoopCompilerContextRequiredError",
    "GroupByMapping",
    "Hierarchy",
    "ThreadGroup",
    "ThreadHierarchy",
    "cpp_level_expr",
    "make_thread_group",
    "normalize_thread_dim",
    "normalize_thread_group_kind",
    "normalize_thread_level",
    "render_group_decl",
    "render_group_decl_lines",
    "render_hierarchy_decl",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    "validate_thread_group_query_dtype",
]
