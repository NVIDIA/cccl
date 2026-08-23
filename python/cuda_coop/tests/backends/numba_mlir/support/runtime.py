# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.models as mlir_models
from numba_cuda_mlir import types
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.extending import lowering_registry
from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir.models import register_model as register_mlir_model
from numba_cuda_mlir.numba_cuda.extending import (
    make_attribute_wrapper,
    type_callable,
    typeof_impl,
)
from numba_cuda_mlir.numba_cuda.extending import models as cuda_models
from numba_cuda_mlir.numba_cuda.extending import (
    register_model as register_cuda_model,
)

import cuda.coop.numba_mlir as coop

_CUDA_RECORD_MODEL = getattr(cuda_models, "Struct" + "Model")
_MLIR_RECORD_MODEL = getattr(mlir_models, "Struct" + "Model")

THREADS = 32
ITEMS_PER_THREAD = 2


class NumbaMlirRunningPrefix:
    def __call__(self_ptr, block_aggregate):
        old_prefix = self_ptr[0]
        self_ptr[0] = old_prefix + block_aggregate
        return old_prefix


NUMBA_MLIR_PREFIX_CALLBACK_OP = coop.StatefulFunction(
    NumbaMlirRunningPrefix,
    types.int32,
    name="numba_mlir_running_prefix",
)


class NumbaMlirKeyPair:
    def __init__(self, key, tie):
        self.key = key
        self.tie = tie

    def construct(this):
        this[0] = NumbaMlirKeyPair(types.int32(0), types.int32(0))

    def assign(this, that):
        this[0] = NumbaMlirKeyPair(that[0].key, that[0].tie)


class NumbaMlirKeyPairType(types.Type):
    def __init__(self):
        super().__init__(name="NumbaMlirKeyPair")

    @property
    def key(self):
        return self.__class__


numba_mlir_keypair_type = NumbaMlirKeyPairType()


@typeof_impl.register(NumbaMlirKeyPair)
def typeof_numba_mlir_keypair(val, c):
    return numba_mlir_keypair_type


@type_callable(NumbaMlirKeyPair)
def type_numba_mlir_keypair(context):
    def typer(key, tie):
        if isinstance(key, types.Integer) and isinstance(tie, types.Integer):
            return numba_mlir_keypair_type

    return typer


@register_cuda_model(NumbaMlirKeyPairType)
class NumbaMlirKeyPairCudaRecordModel(_CUDA_RECORD_MODEL):
    def __init__(self, dmm, fe_type):
        members = [("key", types.int32), ("tie", types.int32)]
        super().__init__(dmm, fe_type, members)


@register_mlir_model(NumbaMlirKeyPairType)
class NumbaMlirKeyPairMlirRecordModel(_MLIR_RECORD_MODEL):
    def __init__(self, dmm, fe_type):
        members = [("key", types.int32), ("tie", types.int32)]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(NumbaMlirKeyPairType, "key", "key")
make_attribute_wrapper(NumbaMlirKeyPairType, "tie", "tie")


@lowering_registry.lower(NumbaMlirKeyPair, types.Integer, types.Integer)
def lower_numba_mlir_keypair(builder, target, args, kwargs):
    key = builder.load_var(args[0])
    tie = builder.load_var(args[1])
    i32 = mlir_ir.IntegerType.get_signless(32)
    key = convert(key, i32)
    tie = convert(tie, i32)
    struct_ty = builder.get_mlir_type(numba_mlir_keypair_type)
    undef = llvm.UndefOp(struct_ty)
    with_key = llvm.insertvalue(
        container=undef,
        value=key,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    result = llvm.insertvalue(
        container=with_key,
        value=tie,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    builder.store_var(target, result)


def _add(a, b):
    return a + b


@cuda.jit(device=True)
def _prefix_with_block_aggregate(block_aggregate):
    return block_aggregate


@cuda.jit(device=True)
def _less(a, b):
    return a < b


@cuda.jit(device=True)
def _complex_real_greater(a, b):
    return a.real > b.real


@cuda.jit(device=True)
def _subtract(a, b):
    return a - b


@cuda.jit(device=True)
def _different(a, b):
    return a != b
