# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Select the shared planner for an operation's semantics type.

Each operation family registers how to classify its parameters and how
to build a lowering plan. Dispatch uses the exact semantics type, so a
new family can supply its rules without adding operation-specific
branches here.

Before calling the family planner, check its accepted group kinds,
required launch capabilities, and resolved group shape. The result is a
plan or a structured unsupported reason. This stage does not compile
code or execute a kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .._types import ParameterClassification
from ..launch import LaunchFacts
from ..thread_group import THREAD_GROUP_KINDS, ThreadGroup
from ._execution_requirements import _unsupported
from ._model import (
    GroupLoweringPlan,
    GroupOperationSemantics,
    GroupPrimitiveCall,
    UnsupportedReasonCode,
)
from ._resolution import _resolve_group


@dataclass(frozen=True)
class _GroupOperationFamily:
    """Keep the planning rules shared by one operation semantics type.

    The registry uses this record to route a ``GroupPrimitiveCall`` to its
    family planner. The callback chooses the implementation and its
    requirements after dispatch has checked the group and launch facts.

    Parameters
    ----------
    classifications : callable
        Map an operation's semantics to its parameter classifications.
        These classifications separate specialization choices, runtime
        arguments, and storage requirements in the call description.
    planner : callable
        Accept the call, resolved group, launch facts, and operation
        semantics. Return a supported or unsupported ``GroupLoweringPlan``.
    group_kinds : frozenset of str
        Group kinds this family can consider. Membership permits planning;
        the family can still reject an unsupported algorithm or shape.
    unsupported_group_message : str
        Diagnostic to use when a call has a group kind outside that set.
    """

    classifications: Callable[
        [GroupOperationSemantics], tuple[ParameterClassification, ...]
    ]
    planner: Callable[
        [GroupPrimitiveCall, ThreadGroup, LaunchFacts, GroupOperationSemantics],
        GroupLoweringPlan,
    ]
    group_kinds: frozenset[str]
    unsupported_group_message: str

    def __post_init__(self) -> None:
        """Reject incomplete registrations before dispatch uses them."""

        if not callable(self.classifications):
            raise TypeError("classifications must be callable")
        if not callable(self.planner):
            raise TypeError("planner must be callable")
        object.__setattr__(self, "group_kinds", frozenset(self.group_kinds))
        if not self.group_kinds:
            raise ValueError("group_kinds must not be empty")
        invalid_group_kinds = {
            kind
            for kind in self.group_kinds
            if not isinstance(kind, str) or kind not in THREAD_GROUP_KINDS
        }
        if invalid_group_kinds:
            names = ", ".join(
                sorted(repr(kind) for kind in invalid_group_kinds)
            )
            raise ValueError(
                f"group_kinds contains unsupported values: {names}"
            )
        if (
            not isinstance(self.unsupported_group_message, str)
            or not self.unsupported_group_message.strip()
        ):
            raise ValueError(
                "unsupported_group_message must be a non-empty string"
            )


_GROUP_OPERATION_FAMILIES: dict[type, _GroupOperationFamily] = {}


def _register_group_operation_family(
    semantics_type: type,
    *,
    classifications: Callable[
        [GroupOperationSemantics], tuple[ParameterClassification, ...]
    ],
    planner: Callable[
        [GroupPrimitiveCall, ThreadGroup, LaunchFacts, GroupOperationSemantics],
        GroupLoweringPlan,
    ],
    group_kinds: frozenset[str],
    unsupported_group_message: str,
) -> None:
    """Install the callbacks for one exact operation semantics type.

    A family module calls this when it loads. Repeating the same registration
    is harmless; replacing it with different rules raises ``RuntimeError``.
    This prevents import order from changing how an existing call is planned.
    See ``_GroupOperationFamily`` for callbacks and group-kind checks.
    """

    if not isinstance(semantics_type, type):
        raise TypeError("semantics_type must be a type")
    registration = _GroupOperationFamily(
        classifications=classifications,
        planner=planner,
        group_kinds=frozenset(group_kinds),
        unsupported_group_message=unsupported_group_message,
    )
    existing = _GROUP_OPERATION_FAMILIES.get(semantics_type)
    if existing is not None and existing != registration:
        raise RuntimeError(
            f"group operation semantics {semantics_type!r} "
            "are already registered"
        )
    _GROUP_OPERATION_FAMILIES[semantics_type] = registration


def _group_operation_family(operation: object) -> _GroupOperationFamily | None:
    """Find the exact semantics type; ignore rules for its base classes."""

    return _GROUP_OPERATION_FAMILIES.get(type(operation))


def _is_group_operation(operation: object) -> bool:
    return _group_operation_family(operation) is not None


def _call_classifications(
    operation: GroupOperationSemantics,
) -> tuple[ParameterClassification, ...]:
    """Ask the registered family which roles its operation parameters have.

    Reject unregistered semantics before constructing a call identity.
    """

    family = _group_operation_family(operation)
    if family is None:
        raise TypeError("unsupported GroupPrimitiveCall operation")
    return family.classifications(operation)


def make_group_primitive_call(
    group: ThreadGroup,
    operation: GroupOperationSemantics,
) -> GroupPrimitiveCall:
    """Pair a group request with the operation semantics to plan.

    ``GroupPrimitiveCall`` validates the descriptor and registered semantics.
    Launch-dependent resolution happens later in ``plan_group_primitive``.
    """

    return GroupPrimitiveCall(group=group, operation=operation)


def plan_group_primitive(
    call: GroupPrimitiveCall,
    launch: LaunchFacts,
) -> GroupLoweringPlan:
    """Resolve a group call and ask its family to choose an implementation.

    Check the family's group-kind limit first. Multi-block cluster operations
    need verified cluster-launch support, and grid operations need verified
    cooperative-launch support. These checks protect implementations whose
    threads must participate together; a shape alone does not establish the
    required launch capability.

    Then resolve the group dimensions and pass the call to its registered
    planner. Family-specific checks choose a supported implementation or
    return a reason that explains why the request cannot be lowered.

    Parameters
    ----------
    call : GroupPrimitiveCall
        Group and operation semantics to plan.
    launch : LaunchFacts
        Exact launch dimensions and capability evidence.

    Returns
    -------
    GroupLoweringPlan
        Selected implementation and execution requirements, or an
        unsupported plan with the failed requirement. The caller decides
        when to turn an unsupported result into a user-facing error.

    Raises
    ------
    TypeError
        An argument has the wrong type or the semantics type is unregistered.
    ValueError
        Group dimensions contradict the launch, or an operation-specific
        argument is invalid.
    """

    if not isinstance(call, GroupPrimitiveCall):
        raise TypeError("call must be a GroupPrimitiveCall")
    if not isinstance(launch, LaunchFacts):
        raise TypeError("launch must be LaunchFacts")
    family = _group_operation_family(call.operation)
    if family is None:
        raise TypeError("unsupported GroupPrimitiveCall operation")
    if call.group.kind not in family.group_kinds:
        return _unsupported(
            call,
            call.group,
            UnsupportedReasonCode.GROUP_KIND,
            family.unsupported_group_message,
        )
    cluster_dim = launch.exact_cluster_dim
    uses_multi_block_cluster = cluster_dim is not None and cluster_dim != (
        1,
        1,
        1,
    )
    if (
        call.group.kind in {"cluster", "grid"}
        and uses_multi_block_cluster
        and (
            launch.cluster_launch is not True
            or not launch.is_verified("cluster_launch")
        )
    ):
        return _unsupported(
            call,
            call.group,
            UnsupportedReasonCode.LAUNCH_CAPABILITY,
            "multi-block cluster lowering requires verified cluster launch "
            f"capability; observed {launch.cluster_launch!r} with verified="
            f"{launch.is_verified('cluster_launch')!r}",
        )
    if call.group.kind == "grid" and (
        launch.cooperative_launch is not True
        or not launch.is_verified("cooperative_launch")
    ):
        return _unsupported(
            call,
            call.group,
            UnsupportedReasonCode.LAUNCH_CAPABILITY,
            "grid group lowering requires verified cooperative launch "
            f"capability; observed {launch.cooperative_launch!r} with "
            f"verified={launch.is_verified('cooperative_launch')!r}",
        )
    resolved, failure = _resolve_group(call, launch)
    if failure is not None:
        return failure
    return family.planner(call, resolved, launch, call.operation)


__all__ = [
    "GroupOperationSemantics",
    "_call_classifications",
    "_is_group_operation",
    "_register_group_operation_family",
    "make_group_primitive_call",
    "plan_group_primitive",
]
