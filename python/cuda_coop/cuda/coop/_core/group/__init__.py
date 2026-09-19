# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable planning for the initial cooperative primitive families."""

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
    GroupTopologyContract,
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
from .exchange import GroupExchangeMode, GroupExchangeSemantics
from .histogram import GroupHistogramSemantics
from .load_store import (
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
)
from .merge_sort import (
    GroupMergeSortSemantics,
)
from .radix import (
    GroupRadixRankSemantics,
    GroupRadixSortSemantics,
)
from .reduce import GroupReduceSemantics
from .run_length import GroupRunLengthDecodeSemantics
from .scan import GroupScanMode, GroupScanSemantics
from .shuffle import GroupShuffleSemantics
from .topk import GroupTopKSemantics

__all__ = [
    "GroupMergeSortSemantics",
    "ArgumentPrecondition",
    "CudaxCallDescription",
    "CudaxReturnKind",
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
    "GroupRadixRankSemantics",
    "GroupRadixSortSemantics",
    "GroupReduceSemantics",
    "GroupRunLengthDecodeSemantics",
    "GroupScanMode",
    "GroupScanSemantics",
    "GroupShuffleSemantics",
    "GroupTopKSemantics",
    "GroupTopologyContract",
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
