# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Pre-compiled operator for BYOC (Bring Your Own Compiler).
"""

from typing import Hashable, Union

from .._bindings import Op, OpKind
from ..struct import _Struct
from ..types import _TypeDescriptor

# Type alias for arguments that can be either a TypeDescriptor or a gpu_struct class
TypeLike = Union[_TypeDescriptor, type]


def _to_type_descriptor(t: TypeLike) -> _TypeDescriptor:
    """Convert a TypeLike to a TypeDescriptor."""
    if isinstance(t, _TypeDescriptor):
        return t
    if isinstance(t, type) and issubclass(t, _Struct):
        return t._get_type_descriptor()
    raise TypeError(
        f"Expected a TypeDescriptor or gpu_struct class, got {type(t).__name__}"
    )


class _CompiledOp:
    """
    Pre-compiled operator from LTOIR bytecode.

    This allows users to bring their own compiler (BYOC) by providing
    pre-compiled LTOIR rather than relying on Numba for JIT compilation.

    The LTOIR must follow the CCCL ABI convention where all arguments and
    the return value are passed as void pointers:
        extern "C" __device__ void name(void* arg0, void* arg1, ..., void* result)

    Example:
        from cuda.compute import CompiledOp, types

        # Compile C++ to LTOIR using cuda.core
        ltoir = compile_cpp_to_ltoir(Program, cpp_source, arch)

        add_op = CompiledOp(
            ltoir=ltoir,
            name="my_add",
            arg_types=(types.int32, types.int32),
            return_type=types.int32,
        )

        reduce_into(d_in, d_out, add_op, num_items, h_init)
    """

    __slots__ = ["_ltoir", "_name", "_arg_types", "_return_type"]

    def __init__(
        self,
        ltoir: bytes,
        name: str,
        arg_types: tuple[TypeLike, ...],
        return_type: TypeLike,
    ):
        """
        Create a pre-compiled operator from LTOIR bytecode.

        Args:
            ltoir: LTOIR bytecode compiled from C++ source
            name: The symbol name of the device function (must match extern "C" name)
            arg_types: Tuple of type descriptors or gpu_struct classes for input arguments
            return_type: Type descriptor or gpu_struct class for the return value
        """
        if not isinstance(ltoir, bytes):
            raise TypeError(f"ltoir must be bytes, got {type(ltoir).__name__}")
        if not ltoir:
            raise ValueError("ltoir cannot be empty")
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got {type(name).__name__}")
        if not name:
            raise ValueError("name cannot be empty")
        if not isinstance(arg_types, tuple):
            raise TypeError(
                f"arg_types must be a tuple, got {type(arg_types).__name__}"
            )

        # Convert all arg_types to TypeDescriptors
        converted_arg_types = []
        for i, arg_type in enumerate(arg_types):
            try:
                converted_arg_types.append(_to_type_descriptor(arg_type))
            except TypeError:
                raise TypeError(
                    f"arg_types[{i}] must be a TypeDescriptor (e.g., types.int32) "
                    f"or a gpu_struct class, got {type(arg_type).__name__}"
                )

        # Convert return_type to TypeDescriptor
        try:
            converted_return_type = _to_type_descriptor(return_type)
        except TypeError:
            raise TypeError(
                f"return_type must be a TypeDescriptor (e.g., types.int32) "
                f"or a gpu_struct class, got {type(return_type).__name__}"
            )

        self._ltoir = ltoir
        self._name = name
        self._arg_types = tuple(converted_arg_types)
        self._return_type = converted_return_type

    def get_cache_key(self) -> Hashable:
        """Return a hashable cache key for this operator."""
        return (hash(self._ltoir), self._name, self._arg_types, self._return_type)

    def compile(self, input_types, output_type=None) -> Op:
        """Compile this operator - returns the pre-compiled Op."""
        return Op(
            operator_type=OpKind.STATELESS,
            name=self._name,
            ltoir=self._ltoir,
            state_alignment=1,
            state=b"",
        )

    @property
    def name(self) -> str:
        """The symbol name of the compiled function."""
        return self._name

    @property
    def ltoir(self) -> bytes:
        """The LTOIR bytecode."""
        return self._ltoir

    @property
    def arg_types(self) -> tuple[_TypeDescriptor, ...]:
        """The argument type descriptors."""
        return self._arg_types

    @property
    def return_type(self) -> _TypeDescriptor:
        """The return type descriptor."""
        return self._return_type

    @property
    def func(self):
        """The underlying callable (None for compiled ops)."""
        return None


# Public alias
CompiledOp = _CompiledOp
