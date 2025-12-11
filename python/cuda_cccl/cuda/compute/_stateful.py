# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING, Callable, List, Tuple

from .typing import DeviceArrayLike

if TYPE_CHECKING:
    pass


def _detect_device_array_globals(func: Callable) -> List[Tuple[str, object]]:
    """
    Detect device arrays referenced as globals in a function.

    Args:
        func: The function to inspect

    Returns:
        List of (name, array) tuples for detected device arrays
    """
    state_arrays = []
    code = func.__code__

    for name in code.co_names:
        val = func.__globals__.get(name)
        if val is not None and isinstance(val, DeviceArrayLike):
            state_arrays.append((name, val))

    return state_arrays


def _detect_device_array_closures(func: Callable) -> List[Tuple[str, object]]:
    """
    Detect device arrays captured in function closures.

    Args:
        func: The function to inspect

    Returns:
        List of (name, array) tuples for detected device arrays
    """
    state_arrays: List[Tuple[str, object]] = []
    code = func.__code__
    closure = func.__closure__

    if closure is None:
        return state_arrays

    # co_freevars contains the names of closure variables
    for name, cell in zip(code.co_freevars, closure):
        try:
            val = cell.cell_contents
            if isinstance(val, DeviceArrayLike):
                state_arrays.append((name, val))
        except ValueError:
            # Cell is empty
            pass

    return state_arrays


def _detect_all_device_arrays(func: Callable) -> List[Tuple[str, object]]:
    """
    Detect all device arrays referenced by a function (globals + closures).

    Args:
        func: The function to inspect

    Returns:
        List of (name, array) tuples for detected device arrays
    """
    globals_arrays = _detect_device_array_globals(func)
    closure_arrays = _detect_device_array_closures(func)
    return globals_arrays + closure_arrays


class _AddStateParameters(ast.NodeTransformer):
    """AST transformer that adds state parameters to a function definition."""

    def __init__(self, state_names: List[str]):
        self.state_names = state_names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Append state parameters to the function arguments (states come last)
        # Inner function signature: (regular_args..., state_arrays...)
        new_args = [ast.arg(arg=name, annotation=None) for name in self.state_names]
        node.args.args = node.args.args + new_args
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        # Handle async functions the same way
        new_args = [ast.arg(arg=name, annotation=None) for name in self.state_names]
        node.args.args = node.args.args + new_args
        return node


def _transform_function_ast(func: Callable, state_names: List[str]) -> Callable:
    """
    Transform a function to add state arrays captured as globals or closures
    as explicit parameters.

    For example, if the function is:

        def func(x): return x + state[0]  # state is a global array

    Then the transformed function will be:

        def func(x, state): return x + state[0]

    Args:
        func: The original function
        state_names: Names of device arrays to add as parameters

    Returns:
        A new function with state arrays as explicit parameters,
        appearing after the regular parameters.

    Raises:
        ValueError: If the function source cannot be obtained
    """
    # Get source code
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as e:
        raise ValueError(
            f"Cannot get source code for function '{func.__name__}'. "
        ) from e

    # Dedent source (in case function is defined inside a class/function)
    source = textwrap.dedent(source)

    # Parse to AST
    tree = ast.parse(source)

    # Transform: add state parameters
    transformer = _AddStateParameters(state_names)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    # Compile and execute to create new function
    # We need to provide the original function's globals for imports
    # AND inject closure variables so they're accessible in the new function
    globals_dict = func.__globals__.copy()

    # Inject closure variables (except state arrays which become parameters)
    if func.__closure__:
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            if name not in state_names:  # Don't inject state arrays
                try:
                    globals_dict[name] = cell.cell_contents
                except ValueError:
                    pass  # Cell is empty

    local_ns: dict[str, Callable] = {}
    exec(
        compile(tree, filename=f"<auto_stateful:{func.__name__}>", mode="exec"),
        globals_dict,
        local_ns,
    )

    # Get the transformed function
    transformed_func = local_ns[func.__name__]

    return transformed_func


def maybe_transform_to_stateful(
    func: Callable,
) -> Tuple[Callable, List[DeviceArrayLike]]:
    """
    If the provided function references device arrays as globals or closures,
    transform it to a function that takes the device arrays as explicit parameters.
    Otherwise, return the original function unchanged.

    Args:
        func: The function to inspect

    Returns:
        A tuple containing the transformed function and the list of device arrays
    """
    # Detect device arrays
    state_info = _detect_all_device_arrays(func)

    if not state_info:
        # No device arrays found, return original
        return func, []

    # Extract names and arrays
    state_names = [name for name, _ in state_info]
    state_arrays: List[DeviceArrayLike] = [arr for _, arr in state_info]  # type: ignore[misc]

    # Transform the function to add state parameters
    transformed_func = _transform_function_ast(func, state_names)

    return transformed_func, state_arrays


__all__ = ["maybe_transform_to_stateful"]
