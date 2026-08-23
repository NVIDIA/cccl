# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Adversarial extension dtypes used to qualify fail-closed CUB storage."""

from numba_cuda_mlir import types
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.models import PrimitiveModel, StructModel, register_model
from numba_cuda_mlir.numba_cuda.extending import models as cuda_models
from numba_cuda_mlir.numba_cuda.extending import (
    register_model as register_cuda_model,
)

_CUDA_STRUCT_MODEL = getattr(cuda_models, "Struct" + "Model")


class PaddedOpaqueType(types.Type):
    """Extension value stored as a padded ``!llvm.struct<(i8, i64)>``."""

    def __init__(self):
        super().__init__(name="PaddedOpaque")

    @property
    def key(self):
        return self.__class__


padded_opaque_type = PaddedOpaqueType()


@register_cuda_model(PaddedOpaqueType)
class PaddedOpaqueCudaModel(_CUDA_STRUCT_MODEL):
    """Present a deceptively inspectable but incompatible CUDA layout."""

    def __init__(self, dmm, fe_type):
        members = [("left", types.int32), ("right", types.int32)]
        super().__init__(dmm, fe_type, members)


@register_model(PaddedOpaqueType)
class PaddedOpaqueModel(PrimitiveModel):
    """Keep the padded LLVM layout opaque to structural ABI inspection."""

    def __init__(self, dmm, fe_type):
        i8 = mlir_ir.IntegerType.get_signless(8)
        i64 = mlir_ir.IntegerType.get_signless(64)
        super().__init__(dmm, fe_type, llvm.StructType.get_literal([i8, i64]))


class MismatchedStructType(types.Type):
    """Extension type whose CUDA and MLIR structural models disagree."""

    def __init__(self):
        super().__init__(name="MismatchedStruct")

    @property
    def key(self):
        return self.__class__


mismatched_struct_type = MismatchedStructType()


@register_cuda_model(MismatchedStructType)
class MismatchedCudaModel(_CUDA_STRUCT_MODEL):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("value", types.uint8)])


@register_model(MismatchedStructType)
class MismatchedMlirModel(StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("value", types.int64)])
