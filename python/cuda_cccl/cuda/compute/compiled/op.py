# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Pre-compiled operator for BYOC (Bring Your Own Compiler).
"""

from typing import Hashable

from .._bindings import Op, OpKind


class _CompiledOp:
    """
    Pre-compiled operator from LTOIR bytecode.

    This allows users to bring their own compiler (BYOC) by providing
    pre-compiled LTOIR rather than relying on Numba for JIT compilation.

    The LTOIR must follow the CCCL ABI convention where all arguments and
    the return value are passed as void pointers:
        extern "C" __device__ void name(void* arg0, void* arg1, ..., void* result)

    Example:
        from cuda.compute import CompiledOp
        from cuda.core import Device, Program, ProgramOptions

        # Compile C++ to LTOIR using cuda.core
        source = '''
        extern "C" __device__ void my_add(void* a, void* b, void* result) {
            *static_cast<int*>(result) = *static_cast<int*>(a) + *static_cast<int*>(b);
        }
        '''
        opts = ProgramOptions(arch="sm_80", relocatable_device_code=True,
                              link_time_optimization=True)
        ltoir = Program(source, "c++", options=opts).compile("ltoir").code

        add_op = CompiledOp(ltoir, "my_add")
        reduce_into(d_in, d_out, add_op, num_items, h_init)
    """

    __slots__ = ["_ltoir", "_name"]

    def __init__(self, ltoir: bytes, name: str):
        """
        Create a pre-compiled operator from LTOIR bytecode.

        Args:
            ltoir: LTOIR bytecode compiled from C++ source
            name: The symbol name of the device function (must match extern "C" name)
        """
        if not isinstance(ltoir, bytes):
            raise TypeError(f"ltoir must be bytes, got {type(ltoir).__name__}")
        if not ltoir:
            raise ValueError("ltoir cannot be empty")
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got {type(name).__name__}")
        if not name:
            raise ValueError("name cannot be empty")

        self._ltoir = ltoir
        self._name = name

    def get_cache_key(self) -> Hashable:
        """Return a hashable cache key for this operator."""
        return (hash(self._ltoir), self._name)

    def compile(self, input_types=None, output_type=None) -> Op:
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
    def func(self):
        """The underlying callable (None for compiled ops)."""
        return None


# Public alias
CompiledOp = _CompiledOp
