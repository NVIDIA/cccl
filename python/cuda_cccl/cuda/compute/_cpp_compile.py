# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
C++ code generation and compilation infrastructure.
"""

from __future__ import annotations

import functools

from cuda.cccl import get_include_paths


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
    arch: str | None = None,
) -> bytes:
    """
    Compile C++ source code to LTOIR.

    Args:
        source: C++ source code string
        arch: Target architecture (e.g., "sm_80"). If None, uses current device.

    Returns:
        LTOIR bytes

    Example:
        source = '''
        extern "C" __device__ void my_add(void* a, void* b, void* result) {
            *static_cast<int*>(result) = *static_cast<int*>(a) + *static_cast<int*>(b);
        }
        '''
        ltoir = compile_cpp_to_ltoir(source)
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


def cpp_type_from_descriptor(type_desc) -> str | None:
    """
    Get the C++ type name from a TypeDescriptor.
    """
    # Careful!!!
    # For efficiency, this function returns None for non-primitive types.
    # Callers must take care to handle that case appropriately.
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

    # STORAGE types have no type name
    return None


def make_variable_declaration(type_desc, name: str) -> str:
    """
    Generate a C++ variable declaration, like "int32_t temp;"
    or "alignas(8) char temp[16];"
    """
    cpp_type = cpp_type_from_descriptor(type_desc)
    if cpp_type is not None:
        return f"{cpp_type} {name};"
    # STORAGE type - use aligned char array
    return f"alignas({type_desc.alignment}) char {name}[{type_desc.size}];"
