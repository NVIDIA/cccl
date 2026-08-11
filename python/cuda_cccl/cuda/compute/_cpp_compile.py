# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
C++ code generation and compilation infrastructure.
"""

from __future__ import annotations

import functools

from cuda.core import Device, Program, ProgramOptions

from cuda.cccl import get_include_paths

from ._bindings import TypeEnum
from ._device_code import DeviceCode

try:
    from ._build_info import USING_V2  # type: ignore[import-not-found]
except ImportError:
    USING_V2 = False


def _get_arch_string() -> str:
    """Target arch string for iterator LTO-IR compilation.

    Honors the build's target compute capability (set for multi-arch / no-GPU
    builds) so iterator device code is compiled for the lowest target arch and
    links into every build result; falls back to the current device otherwise.
    """
    from ._target_cc import get_target_cc

    cc = get_target_cc()
    if cc is None:
        cc = Device().compute_capability
    cc_major, cc_minor = cc
    return f"sm_{cc_major}{cc_minor}"


@functools.lru_cache(maxsize=1)
def _get_include_paths() -> list[str]:
    """Get include paths for CCCL headers."""
    paths = get_include_paths().as_tuple()
    return [p for p in paths if p is not None]


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
    # Resolve the concrete arch before the cache lookup so the key reflects the
    # compute capability compiled for. If arch stays None (the usual iterator/op
    # call, resolved from target_cc), every target collapses to one key and
    # LTO-IR built for one arch can be reused for another, which nvJitLink
    # rejects.
    if arch is None:
        arch = _get_arch_string()
    return _compile_cpp_to_ltoir_cached(source, arch)


@functools.lru_cache(maxsize=256)
def _compile_cpp_to_ltoir_cached(source: str, arch: str) -> bytes:
    # Get include paths
    include_paths = _get_include_paths()

    # Configure compilation options for LTO
    opts = ProgramOptions(
        arch=arch,
        relocatable_device_code=True,
        link_time_optimization=True,
        std="c++20",
        define_macro="__NV_NO_VECTOR_DEPRECATION_DIAG",
        include_path=include_paths,
    )

    # Compile to LTOIR
    program = Program(source, "c++", options=opts)
    result = program.compile("ltoir")

    return result.code


# Expose the cached-callable surface (cache_info/cache_clear) on the public
# entry point, backed by the arch-aware inner cache.
compile_cpp_to_ltoir.cache_clear = _compile_cpp_to_ltoir_cached.cache_clear  # type: ignore[attr-defined]
compile_cpp_to_ltoir.cache_info = _compile_cpp_to_ltoir_cached.cache_info  # type: ignore[attr-defined]


def compile_cpp_op_code(source: str, arch: str | None = None) -> DeviceCode:
    """Compile C++ wrapper source to whatever form the active backend prefers.

    Returns a :class:`DeviceCode` wrapping the bytes and the matching format tag.

    Cached so identical iterator structures produce identical code bytes —
    callers can inspect ``cache_info()`` to verify symbol determinism.
    """
    # v2 keeps the C++ source verbatim (arch-independent); v1 resolves the
    # concrete arch before caching (see compile_cpp_to_ltoir).
    if USING_V2:
        return _compile_cpp_op_code_cached(source, None)
    if arch is None:
        arch = _get_arch_string()
    return _compile_cpp_op_code_cached(source, arch)


@functools.lru_cache(maxsize=256)
def _compile_cpp_op_code_cached(source: str, arch: str | None) -> DeviceCode:
    if USING_V2:
        return DeviceCode(op_bytes=source.encode("utf-8"), kind="cpp_source")
    return DeviceCode(op_bytes=compile_cpp_to_ltoir(source, arch=arch), kind="ltoir")


compile_cpp_op_code.cache_clear = _compile_cpp_op_code_cached.cache_clear  # type: ignore[attr-defined]
compile_cpp_op_code.cache_info = _compile_cpp_op_code_cached.cache_info  # type: ignore[attr-defined]


def cpp_type_from_descriptor(type_desc) -> str | None:
    """
    Get the C++ type name from a TypeDescriptor.

    Important: for efficiency, this function returns None
    for non-primitive types. Callers must take care
    to handle that case appropriately.
    """
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
        TypeEnum.STORAGE: None,
    }
    return type_map[type_desc.info.typenum]


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
