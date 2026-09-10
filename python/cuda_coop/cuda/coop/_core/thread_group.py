# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUDA thread-hierarchy and group descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Any, TypeVar

# Hierarchy levels identify the coordinate spaces used by group descriptors.
THREAD_LEVELS = frozenset({"thread", "warp", "block", "cluster", "grid"})
# Physical group kinds identify runtime execution groups rather than mapped groups.
PHYSICAL_GROUP_KINDS = frozenset({"thread", "warp", "block", "cluster", "grid"})
MAPPED_GROUP_KINDS = frozenset({"threads_within_warp", "warps_within_block"})
THREAD_GROUP_KINDS = PHYSICAL_GROUP_KINDS | MAPPED_GROUP_KINDS
COMPLETE_WARP_GROUP_KINDS = frozenset({"warp"}) | MAPPED_GROUP_KINDS
_ThreadGroupT = TypeVar("_ThreadGroupT", bound="ThreadGroup")
_CPP_LEVEL_EXPR = {
    "thread": "::cuda::gpu_thread",
    "warp": "::cuda::warp",
    "block": "::cuda::block",
    "cluster": "::cuda::cluster",
    "grid": "::cuda::grid",
}


class CoopCompilerContextRequiredError(RuntimeError):
    """A compiler-facing cooperative value escaped its compiler context."""


def normalize_thread_dim(
    value: Any,
    *,
    scope: str,
    label: str,
) -> tuple[int, int, int]:
    """Normalize a one-, two-, or three-dimensional CUDA launch shape."""

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
    """Return the canonical spelling for a CUDA hierarchy level."""

    if level == "gpu_thread":
        level = "thread"
    if level not in THREAD_LEVELS:
        names = ", ".join(sorted(THREAD_LEVELS))
        raise ValueError(f"{scope}.{feature} level must be one of: {names}")
    return level


def normalize_thread_group_kind(kind: str, *, scope: str, feature: str) -> str:
    """Return a canonical physical or statically mapped group kind."""

    if kind == "gpu_thread":
        kind = "thread"
    if kind not in THREAD_GROUP_KINDS:
        names = ", ".join(sorted(THREAD_GROUP_KINDS))
        raise ValueError(f"{scope}.{feature} group kind must be one of: {names}")
    return kind


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
    """CUDA hierarchy descriptor for the current kernel launch.

    Public construction always denotes the active launch. Backends resolve
    exact extents from verified :class:`LaunchFacts`; callers cannot assert
    launch dimensions independently of the compiler.
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
    ) -> "ThreadHierarchy":
        """Materialize planner-verified extents from launch facts."""

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
    def current(cls) -> "ThreadHierarchy":
        """Describe C++ default ``this_*()`` hierarchy construction."""

        return cls()

    @property
    def is_static(self) -> bool:
        return not self.implicit

    @property
    def block_thread_count(self) -> int | None:
        """Return the enclosing CTA size when it is statically known."""

        return _thread_count(self.block_dim)  # type: ignore[arg-type]

    @property
    def symbol_suffix(self) -> str:
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
        """Return the canonical symbol token for this hierarchy's block shape."""

        if self.block_dim is None:
            return "current"
        return _dims_token("b", self.block_dim)

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.block_dim, self.grid_dim, self.cluster_dim, self.implicit

    def has_static_extents_for(self, group_kind: str) -> bool:
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
    """Static CUDAX ``group_by`` mapping semantics."""

    unit: str
    parent: str
    count: int
    exhaustive: bool
    synchronizer: str

    def __post_init__(self) -> None:
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
    """Backend-neutral descriptor for one group in a CUDA hierarchy."""

    kind: str
    hierarchy: ThreadHierarchy = field(default_factory=ThreadHierarchy.current)
    parent: "ThreadGroup | None" = None
    mapping: GroupByMapping | None = None
    # Provenance is excluded from semantic identity and cache keys, but planners
    # may still use it to preserve policy at public API boundaries.
    source: str = field(default="explicit", compare=False, hash=False)

    def __post_init__(self) -> None:
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
        """Return the planner-resolved enclosing block dimensions, if known."""

        return self.hierarchy.block_dim

    @property
    def group_thread_count(self) -> int:
        """Return the number of threads in this group.

        The physical warp extent is always 32. Whether all 32 lanes are valid
        participants for a particular collective is a lowering-legality
        question and is deliberately not encoded by this descriptor.
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
        """Return the static group extent independently of its parent CTA."""

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
        return self.hierarchy.implicit  # type: ignore[union-attr]

    @property
    def is_static(self) -> bool:
        """Whether all hierarchy extents required by this group are static."""

        if self.kind in MAPPED_GROUP_KINDS:
            assert self.parent is not None
            return self.parent.is_static
        return self.hierarchy.has_static_extents_for(self.kind)  # type: ignore[union-attr]

    @property
    def parent_unit_count(self) -> int | None:
        """Return the static number of mapped units in the parent group."""

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
        """Return the number of complete mapped groups in one parent."""

        if self.mapping is None:
            return None
        parent_units = self.parent_unit_count
        if parent_units is None:
            return None
        return parent_units // self.mapping.count

    @property
    def remainder_count(self) -> int | None:
        """Return mapped parent units excluded by a non-exhaustive mapping."""

        if self.mapping is None:
            return None
        parent_units = self.parent_unit_count
        if parent_units is None:
            return None
        return parent_units % self.mapping.count

    @property
    def complete_membership(self) -> bool | None:
        """Whether the mapping covers every unit in its physical parent."""

        if self.mapping is None:
            return True
        remainder = self.remainder_count
        if remainder is None:
            return None
        return remainder == 0

    @property
    def symbol_suffix(self) -> str:
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
        if self.mapping is None:
            return self.kind, self.hierarchy.semantic_key  # type: ignore[union-attr]
        assert self.parent is not None
        return (
            self.kind,
            self.parent.semantic_key,
            self.mapping.semantic_key,
            self.hierarchy.semantic_key,  # type: ignore[union-attr]
        )

    def with_hierarchy(
        self: _ThreadGroupT,
        hierarchy: ThreadHierarchy,
        *,
        source: str = "resolved",
    ) -> _ThreadGroupT:
        """Return the same backend group type with resolved hierarchy extents."""

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

    def group_by(
        self: _ThreadGroupT,
        count: int,
        *,
        exhaustive: bool = True,
    ) -> _ThreadGroupT:
        """Partition a physical warp by threads or a block by warps."""

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


def make_thread_group(
    kind: str,
    *,
    group_type: type[_ThreadGroupT] = ThreadGroup,
    scope: str = "cuda.coop",
) -> _ThreadGroupT:
    """Build a current-launch group using a backend-selected group type."""

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
    """Render one static CUDAX hierarchy declaration."""

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
    """Render one physical CUDAX group declaration."""

    if group.mapping is not None:
        raise ValueError(
            "render_group_decl supports physical groups only; use "
            "render_group_decl_lines for mapped groups"
        )
    assert group.hierarchy is not None
    if group.hierarchy.implicit:
        return (
            f"{indent}::cuda::experimental::this_{group.kind} "
            f"{var_name}{{::cuda::experimental::implicit_hierarchy()}};"
        )
    return (
        f"{indent}::cuda::experimental::this_{group.kind} "
        f"{var_name}{{{hierarchy_var}}};"
    )


def render_group_decl_lines(
    group: ThreadGroup,
    *,
    var_name: str = "group",
    hierarchy_var: str = "hierarchy",
    indent: str = "  ",
) -> list[str]:
    """Render a physical or statically mapped CUDAX group declaration."""

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
                f"{indent}::cuda::experimental::group {var_name}{{",
                f"{indent}    ::cuda::gpu_thread, {parent_name},",
                f"{indent}    ::cuda::experimental::group_by<"
                f"{mapping.count}, {exhaustive}>{{}},",
                f"{indent}    ::cuda::experimental::lane_synchronizer{{}}}};",
            ]
        )
        return lines

    groups_per_parent = group.groups_per_parent
    if groups_per_parent is None:
        raise ValueError("mapped warp group requires a static parent group count")
    lines.extend(
        [
            f"{indent}using {var_name}_barriers_type =",
            f"{indent}    ::cuda::barrier<::cuda::thread_scope_block>"
            f"[{groups_per_parent}];",
            f"{indent}__shared__ ::cuda::std::aligned_storage_t<",
            f"{indent}    sizeof({var_name}_barriers_type),",
            f"{indent}    alignof({var_name}_barriers_type)> "
            f"{var_name}_barriers_storage;",
            f"{indent}auto& {var_name}_barriers =",
            f"{indent}    reinterpret_cast<{var_name}_barriers_type&>(",
            f"{indent}        {var_name}_barriers_storage);",
            f"{indent}::cuda::experimental::group {var_name}{{",
            f"{indent}    ::cuda::warp, {parent_name},",
            f"{indent}    ::cuda::experimental::group_by<"
            f"{mapping.count}, {exhaustive}>{{}},",
            f"{indent}    ::cuda::experimental::barrier_synchronizer{{"
            f"{var_name}_barriers}}}};",
        ]
    )
    return lines


def cpp_level_expr(level: str) -> str:
    """Return the CUDAX hierarchy-level expression for ``level``."""

    return _CPP_LEVEL_EXPR[
        normalize_thread_level(level, scope="cuda.coop", feature="cpp_level_expr")
    ]


def this_thread() -> ThreadGroup:
    return make_thread_group("thread")


def this_warp() -> ThreadGroup:
    return make_thread_group("warp")


def this_block() -> ThreadGroup:
    return make_thread_group("block")


def this_cluster() -> ThreadGroup:
    return make_thread_group("cluster")


def this_grid() -> ThreadGroup:
    return make_thread_group("grid")


__all__ = [
    "COMPLETE_WARP_GROUP_KINDS",
    "CoopCompilerContextRequiredError",
    "MAPPED_GROUP_KINDS",
    "PHYSICAL_GROUP_KINDS",
    "THREAD_LEVELS",
    "THREAD_GROUP_KINDS",
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
]
