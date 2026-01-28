# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba intrinsics and intrinsic factory functions for CUDA compute operations."""

import operator

import numba
from llvmlite import ir
from numba import types
from numba.core.extending import intrinsic, overload

_DEVICE_POINTER_SIZE = 8
_DEVICE_POINTER_BITWIDTH = _DEVICE_POINTER_SIZE * 8


def make_alloca_intrinsic(type_descriptor):
    """Create an intrinsic for allocating temporary storage of the given type.

    This factory creates a Numba intrinsic that allocates stack storage for
    a value of the specified type and returns a pointer to it.

    Args:
        type_descriptor: A TypeDescriptor (including PointerTypeDescriptor)

    Returns:
        An intrinsic function that, when called, returns a CPointer to
        allocated storage of the given type.

    Example:
        from cuda.compute import types
        alloca_temp = make_alloca_intrinsic(types.float32)
        # In a device function:
        temp_ptr = alloca_temp()  # Returns CPointer(float32)
    """
    from ._jit import type_descriptor_to_numba

    numba_type = type_descriptor_to_numba(type_descriptor)

    @intrinsic
    def alloca_temp(context):
        def codegen(context, builder, sig, args):
            temp_value_type = context.get_value_type(numba_type)
            temp_ptr = builder.alloca(temp_value_type)
            return temp_ptr

        return types.CPointer(numba_type)(), codegen

    return alloca_temp


def make_struct_field_ptr_intrinsic(field_idx: int):
    """Create an intrinsic for accessing a struct field by index.

    This factory creates a Numba intrinsic that takes a pointer to a struct
    and returns a pointer to the field at the specified index.

    Args:
        field_idx: The 0-based index of the field to access

    Returns:
        An intrinsic function that takes a struct pointer and returns
        a CPointer to the specified field.

    Example:
        get_first_field = make_struct_field_ptr_intrinsic(0)
        get_second_field = make_struct_field_ptr_intrinsic(1)
        # In a device function:
        field0_ptr = get_first_field(struct_ptr)  # Returns CPointer to field 0
    """
    from numba.core.datamodel.registry import default_manager

    @intrinsic
    def get_field_ptr(context, struct_ptr_type):
        def codegen(context, builder, sig, args):
            struct_ptr = args[0]
            field_ptr = builder.gep(
                struct_ptr,
                [
                    ir.Constant(ir.IntType(32), 0),
                    ir.Constant(ir.IntType(32), field_idx),
                ],
            )
            return field_ptr

        struct_model = default_manager.lookup(struct_ptr_type.dtype)
        field_type = struct_model._members[field_idx]
        return types.CPointer(field_type)(struct_ptr_type), codegen

    return get_field_ptr


def _sizeof_pointee(context, ptr):
    """Helper to get size of pointee type as an IR constant."""
    size = context.get_abi_sizeof(ptr.type.pointee)
    return ir.Constant(ir.IntType(_DEVICE_POINTER_BITWIDTH), size)


@intrinsic
def pointer_add_intrinsic(context, ptr, offset):
    """Intrinsic for pointer arithmetic (ptr + offset)."""

    def codegen(context, builder, sig, args):
        ptr, index = args
        base = builder.ptrtoint(ptr, ir.IntType(_DEVICE_POINTER_BITWIDTH))
        sizeof = _sizeof_pointee(context, ptr)
        # Cast index to match sizeof type if needed
        if index.type != sizeof.type:
            index = (
                builder.sext(index, sizeof.type)
                if index.type.width < sizeof.type.width
                else builder.trunc(index, sizeof.type)
            )
        offset = builder.mul(index, sizeof)
        result = builder.add(base, offset)
        return builder.inttoptr(result, ptr.type)

    return ptr(ptr, offset), codegen


@overload(operator.add)
def pointer_add(ptr, offset):
    """Overload operator.add for CPointer + Integer."""
    if not isinstance(ptr, numba.types.CPointer) or not isinstance(
        offset, numba.types.Integer
    ):
        return

    def impl(ptr, offset):
        return pointer_add_intrinsic(ptr, offset)

    return impl


@intrinsic
def load_cs(typingctx, base):
    """Load with cache streaming hint (ld.global.cs).

    Corresponding to `LOAD_CS` here:
    https://nvidia.github.io/cccl/cub/api/classcub_1_1CacheModifiedInputIterator.html
    """

    def codegen(context, builder, sig, args):
        rt = context.get_value_type(sig.return_type)
        if rt is None:
            raise RuntimeError(f"Unsupported return type: {type(sig.return_type)}")
        ftype = ir.FunctionType(rt, [rt.as_pointer()])
        bw = sig.return_type.bitwidth
        asm_txt = f"ld.global.cs.b{bw} $0, [$1];"
        if bw < 64:
            constraint = "=r, l"
        else:
            constraint = "=l, l"
        asm_ir = ir.InlineAsm(ftype, asm_txt, constraint)
        return builder.call(asm_ir, args)

    return base.dtype(base), codegen


__all__ = [
    "load_cs",
    "make_alloca_intrinsic",
    "make_struct_field_ptr_intrinsic",
    "pointer_add_intrinsic",
]
