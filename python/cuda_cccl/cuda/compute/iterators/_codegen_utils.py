# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Code generation utilities for iterators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ._base import IteratorBase


# ============================================================================
# State Composition Utilities
# ============================================================================


def compose_iterator_states(
    iterators: list[IteratorBase],
) -> tuple[bytes, int, list[int]]:
    """
    Concatenate multiple iterator states with proper alignment.

    This is used by composite iterators (like ZipIterator and PermutationIterator)
    that need to store multiple child iterator states in their own state.

    Args:
        iterators: List of child iterators whose states should be composed

    Returns:
        Tuple of:
        - combined_state_bytes: Concatenated state bytes with padding
        - combined_alignment: Maximum alignment requirement
        - offsets: List of byte offsets for each iterator's state
    """
    if not iterators:
        return (b"", 1, [])

    states = [bytes(memoryview(it.state)) for it in iterators]
    alignments = [it.state_alignment for it in iterators]

    offsets = []
    current_offset = 0
    combined = b""

    for state, align in zip(states, alignments):
        # Add padding to meet alignment requirement
        padding = (align - (current_offset % align)) % align
        combined += b"\x00" * padding
        current_offset += padding

        offsets.append(current_offset)
        combined += state
        current_offset += len(state)

    max_alignment = max(alignments)
    return (combined, max_alignment, offsets)


CUDA_PREAMBLE = """#include <cuda/std/cstdint>
#include <cuda_fp16.h>
#include <cuda/std/cstring>
using namespace cuda::std;
"""


def make_extern_decl(symbol: str) -> str:
    """Generate an extern C device function declaration."""
    return f'extern "C" __device__ void {symbol}(void*, void*);'


def make_extern_decls(symbols: list[str]) -> str:
    """Generate multiple extern C device function declarations."""
    return "\n".join(make_extern_decl(sym) for sym in symbols)


# ============================================================================
# Standard Operation Templates
# ============================================================================

_ADVANCE_TEMPLATE = """{preamble}

{extern_decls}

extern "C" __device__ void {symbol}(void* state, void* offset) {{
{body}
}}
"""

_INPUT_DEREF_TEMPLATE = """{preamble}

{extern_decls}

extern "C" __device__ void {symbol}(void* state, void* result) {{
{body}
}}
"""

_OUTPUT_DEREF_TEMPLATE = """{preamble}

{extern_decls}

extern "C" __device__ void {symbol}(void* state, void* value) {{
{body}
}}
"""


def _format_template(
    template: str,
    symbol: str,
    body: str,
    extern_symbols: Optional[list[str]] = None,
    preamble: str = CUDA_PREAMBLE,
) -> str:
    """
    Format a C++ code template with the given parameters.

    Args:
        template: Template string with placeholders
        symbol: Name for the generated function
        body: Function body code (will be indented)
        extern_symbols: List of external symbols to declare (optional)
        preamble: C++ includes/using statements (default: CUDA_PREAMBLE)

    Returns:
        Formatted C++ source code
    """
    extern_decls = make_extern_decls(extern_symbols) if extern_symbols else ""

    # Indent body lines (but not empty lines)
    body_lines = body.split("\n")
    indented_body = "\n".join(
        "    " + line if line.strip() else line for line in body_lines
    )

    return template.format(
        preamble=preamble, extern_decls=extern_decls, symbol=symbol, body=indented_body
    )


def format_advance(
    symbol: str,
    body: str,
    extern_symbols: Optional[list[str]] = None,
    preamble: str = CUDA_PREAMBLE,
) -> str:
    """
    Format an advance operation.

    Generates: extern "C" __device__ void symbol(void* state, void* offset)

    Args:
        symbol: Name for the generated function
        body: Function body code (will be indented)
        extern_symbols: List of external symbols to declare (optional)
        preamble: C++ includes/using statements (default: CUDA_PREAMBLE)

    Returns:
        Formatted C++ source code
    """
    return _format_template(_ADVANCE_TEMPLATE, symbol, body, extern_symbols, preamble)


def format_input_dereference(
    symbol: str,
    body: str,
    extern_symbols: Optional[list[str]] = None,
    preamble: str = CUDA_PREAMBLE,
) -> str:
    """
    Format an input dereference operation.

    Generates: extern "C" __device__ void symbol(void* state, void* result)

    Args:
        symbol: Name for the generated function
        body: Function body code (will be indented)
        extern_symbols: List of external symbols to declare (optional)
        preamble: C++ includes/using statements (default: CUDA_PREAMBLE)

    Returns:
        Formatted C++ source code
    """
    return _format_template(
        _INPUT_DEREF_TEMPLATE, symbol, body, extern_symbols, preamble
    )


def format_output_dereference(
    symbol: str,
    body: str,
    extern_symbols: Optional[list[str]] = None,
    preamble: str = CUDA_PREAMBLE,
) -> str:
    """
    Format an output dereference operation.

    Generates: extern "C" __device__ void symbol(void* state, void* value)

    Args:
        symbol: Name for the generated function
        body: Function body code (will be indented)
        extern_symbols: List of external symbols to declare (optional)
        preamble: C++ includes/using statements (default: CUDA_PREAMBLE)

    Returns:
        Formatted C++ source code
    """
    return _format_template(
        _OUTPUT_DEREF_TEMPLATE, symbol, body, extern_symbols, preamble
    )


# ============================================================================
# Compilation Utilities
# ============================================================================


def compile_cpp_source_to_ltoir(source: str, symbol: str) -> bytes:
    """
    Compile C++ source code to LTOIR.

    This is a convenience wrapper around the lower-level compile_cpp_to_ltoir
    function, provided as a utility for iterators that generate C++ source.
    Iterators using other compilation methods can provide LTOIR directly.

    Args:
        source: Complete C++ source code (including preamble, externs, function)
        symbol: Symbol name of the function being compiled

    Returns:
        Compiled LTOIR bytes
    """
    from .._cpp_codegen import compile_cpp_to_ltoir

    return compile_cpp_to_ltoir(source, (symbol,))


def collect_child_ltoirs(children, operation: str) -> list[bytes]:
    """
    Collect LTOIRs from child iterators for a specific operation.

    This is a utility function for composite iterators (Zip, Transform, etc.)
    that need to include their children's LTOIR in their own extra_ltoirs.

    Args:
        children: Iterable of child IteratorBase instances
        operation: Operation type - "advance", "input_deref", or "output_deref"

    Returns:
        List of LTOIR bytes from all children and their transitive dependencies
    """
    extras = []
    for child in children:
        if operation == "advance":
            _, ltoir, child_extras = child.get_advance_ltoir()
            extras.extend([ltoir] + child_extras)
        elif operation == "input_deref":
            result = child.get_input_dereference_ltoir()
            if result is not None:
                _, ltoir, child_extras = result
                extras.extend([ltoir] + child_extras)
        elif operation == "output_deref":
            result = child.get_output_dereference_ltoir()
            if result is not None:
                _, ltoir, child_extras = result
                extras.extend([ltoir] + child_extras)
    return extras
