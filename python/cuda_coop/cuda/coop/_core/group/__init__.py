# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable semantic-family planning for cooperative thread-group primitives.

Each family module owns one primitive's normalized semantics and CUDAX/CUB
selection. Shared model, resolution, contract, and routing modules preserve the
cross-family cache and artifact invariants without depending on a backend.
"""

from ._dispatch import (
    GroupOperationSemantics,
    make_group_primitive_call,
    plan_group_primitive,
)
from ._model import (
    ArgumentPrecondition,
    CudaxCallDescription,
    CudaxReturnKind,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    GroupPrimitiveCall,
    ImplementationProvenance,
    LogicalResultContract,
    ParticipationContract,
    PreconditionEnforcement,
    ResultContract,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    SynchronizationContract,
    SynchronizationScope,
    TempStorageContract,
    ThreadGroupResolution,
    UnsupportedReason,
    UnsupportedReasonCode,
)
from ._resolution import resolve_thread_group
from .adjacent_difference import GroupAdjacentDifferenceSemantics
from .discontinuity import GroupDiscontinuitySemantics
from .exchange import GroupExchangeMode, GroupExchangeSemantics
from .histogram import GroupHistogramSemantics
from .load_store import (
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
)
from .reduce import GroupReduceSemantics
from .run_length_decode import GroupRunLengthDecodeSemantics
from .scan import GroupScanMode, GroupScanSemantics
from .shuffle import GroupShuffleSemantics

__all__ = [
    "ArgumentPrecondition",
    "CudaxCallDescription",
    "CudaxReturnKind",
    "GroupAdjacentDifferenceSemantics",
    "GroupDiscontinuitySemantics",
    "GroupExchangeMode",
    "GroupExchangeSemantics",
    "GroupHistogramSemantics",
    "GroupLoadStoreAlgorithm",
    "GroupLoadStoreKind",
    "GroupLoadStoreSemantics",
    "GroupLoweringPlan",
    "GroupLoweringTarget",
    "GroupOperandKind",
    "GroupOperationSemantics",
    "GroupPrimitiveCall",
    "GroupReduceSemantics",
    "GroupRunLengthDecodeSemantics",
    "GroupScanMode",
    "GroupScanSemantics",
    "GroupShuffleSemantics",
    "ImplementationProvenance",
    "LogicalResultContract",
    "ParticipationContract",
    "PreconditionEnforcement",
    "ResultContract",
    "ResultOwnership",
    "ResultVisibility",
    "StorageOwnership",
    "SynchronizationContract",
    "SynchronizationScope",
    "TempStorageContract",
    "ThreadGroupResolution",
    "UnsupportedReason",
    "UnsupportedReasonCode",
    "make_group_primitive_call",
    "plan_group_primitive",
    "resolve_thread_group",
]
