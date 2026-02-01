# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
C++ code generation and compilation infrastructure for cuda.compute.

This module provides utilities to generate C++ source code and compile it
to LTOIR (Link-Time Optimization IR) for use with CCCL algorithms.
"""

from __future__ import annotations

import functools

from cuda.cccl import get_include_paths  # type: ignore[import-not-found]


def _get_arch_string() -> str:
    """Get the compute capability string for the current device."""
    from cuda.core import Device

    device = Device()
    cc_major, cc_minor = device.compute_capability
    return f"sm_{cc_major}{cc_minor}"


@functools.lru_cache(maxsize=1)
def _get_include_paths() -> list[str]:
    """Get include paths for CCCL headers."""
    paths = get_include_paths().as_tuple()
    return [p for p in paths if p is not None]


@functools.lru_cache(maxsize=256)
def compile_cpp_to_ltoir(
    source: str,
    symbols: tuple[str, ...],
    arch: str | None = None,
) -> bytes:
    """
    Compile C++ source code to LTOIR.

    Args:
        source: C++ source code string
        symbols: Tuple of symbol names to extract from the compiled code
        arch: Target architecture (e.g., "sm_80"). If None, uses current device.

    Returns:
        LTOIR bytes that can be used with CompiledOp

    Example:
        source = '''
        extern "C" __device__ void my_add(void* a, void* b, void* result) {
            *static_cast<int*>(result) = *static_cast<int*>(a) + *static_cast<int*>(b);
        }
        '''
        ltoir = compile_cpp_to_ltoir(source, ("my_add",))
    """
    from cuda.core import Program, ProgramOptions

    if arch is None:
        arch = _get_arch_string()

    # Get include paths
    include_paths = _get_include_paths()

    # Configure compilation options for LTO
    opts = ProgramOptions(
        arch=arch,
        relocatable_device_code=True,
        link_time_optimization=True,
        std="c++17",
        include_path=include_paths,
    )

    # Compile to LTOIR
    program = Program(source, "c++", options=opts)
    result = program.compile("ltoir")

    return result.code


def cpp_type_from_descriptor(type_desc) -> str:
    """
    Get the C++ type name from a TypeDescriptor.

    Args:
        type_desc: A TypeDescriptor instance

    Returns:
        C++ type name string. For struct/storage types, returns an inline
        anonymous struct type like: struct alignas(8) { char _[16]; }
    """
    from ._bindings import TypeEnum

    # Map TypeEnum to C++ types
    type_map = {
        TypeEnum.INT8: "int8_t",
        TypeEnum.INT16: "int16_t",
        TypeEnum.INT32: "int32_t",
        TypeEnum.INT64: "int64_t",
        TypeEnum.UINT8: "uint8_t",
        TypeEnum.UINT16: "uint16_t",
        TypeEnum.UINT32: "uint32_t",
        TypeEnum.UINT64: "uint64_t",
        TypeEnum.FLOAT16: "__half",
        TypeEnum.FLOAT32: "float",
        TypeEnum.FLOAT64: "double",
        TypeEnum.BOOLEAN: "bool",
    }

    if type_desc.info.typenum in type_map:
        return type_map[type_desc.info.typenum]

    # For STORAGE types, return an inline anonymous struct
    size = type_desc.size
    alignment = type_desc.alignment
    return f"struct alignas({alignment}) {{ char _[{size}]; }}"
