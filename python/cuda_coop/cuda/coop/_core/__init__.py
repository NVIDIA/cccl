# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral cooperative API contracts."""

from ._bindings import ArgumentBinding, BindingKind
from .group_dispatch import (
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupPrimitiveCall,
    ImplementationProvenance,
    ParticipationContract,
    ResultContract,
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
    "Dim3",
    "GroupLoadStoreAlgorithm",
    "GroupLoadStoreKind",
    "GroupLoadStoreSemantics",
    "GroupLoweringPlan",
    "GroupLoweringTarget",
    "GroupPrimitiveCall",
    "ImplementationProvenance",
    "LaunchFactOrigin",
    "LaunchFacts",
    "ParticipationContract",
    "ResultContract",
    "StorageOwnership",
    "SynchronizationContract",
    "SynchronizationScope",
    "TempStorageContract",
    "ThreadGroup",
    "ThreadHierarchy",
    "make_group_primitive_call",
    "plan_group_primitive",
    "resolve_thread_group",
    "this_block",
]
