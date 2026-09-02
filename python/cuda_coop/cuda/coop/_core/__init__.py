# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiler-free cooperative reduction contracts."""

from ._bindings import ArgumentBinding, BindingKind
from ._errors import CoopCompilerContextRequiredError
from .block import (
    BlockReduceAlgorithm,
    BlockReduceOperation,
    BlockReduceOperator,
    BlockReduceSpec,
    make_block_reduce_spec,
    normalize_block_reduce_algorithm,
    normalize_block_reduce_operator,
)
from .group import (
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupPrimitiveCall,
    GroupReduceAlgorithm,
    GroupReduceOperation,
    GroupReduceOperator,
    GroupReduceSemantics,
    ImplementationProvenance,
    ParticipationContract,
    ResultContract,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    SynchronizationContract,
    SynchronizationScope,
    TempStorageContract,
    make_group_primitive_call,
    plan_group_primitive,
    resolve_thread_group,
)
from .launch import Dim3, LaunchFactOrigin, LaunchFacts
from .thread_group import ThreadGroup, ThreadHierarchy, this_block

__all__ = [
    "ArgumentBinding",
    "BindingKind",
    "BlockReduceAlgorithm",
    "BlockReduceOperation",
    "BlockReduceOperator",
    "BlockReduceSpec",
    "CoopCompilerContextRequiredError",
    "Dim3",
    "GroupLoweringPlan",
    "GroupLoweringTarget",
    "GroupOperandKind",
    "GroupPrimitiveCall",
    "GroupReduceAlgorithm",
    "GroupReduceOperation",
    "GroupReduceOperator",
    "GroupReduceSemantics",
    "ImplementationProvenance",
    "LaunchFactOrigin",
    "LaunchFacts",
    "ParticipationContract",
    "ResultContract",
    "ResultOwnership",
    "ResultVisibility",
    "StorageOwnership",
    "SynchronizationContract",
    "SynchronizationScope",
    "TempStorageContract",
    "ThreadGroup",
    "ThreadHierarchy",
    "make_block_reduce_spec",
    "make_group_primitive_call",
    "normalize_block_reduce_algorithm",
    "normalize_block_reduce_operator",
    "plan_group_primitive",
    "resolve_thread_group",
    "this_block",
]
