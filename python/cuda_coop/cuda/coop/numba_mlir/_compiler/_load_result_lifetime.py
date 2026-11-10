# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check whether a returning Load would overwrite an earlier live result.

Numba local arrays have one allocation per call site. Repeated execution is
safe when the previous result is dead, including when its variable is simply
reassigned. Follow only payload references and tuples containing references;
reading a payload element produces a scalar, not another reference.
"""

import operator
from collections import deque
from numbers import Integral

from numba_cuda_mlir.numba_cuda.core.analysis import (
    compute_cfg_from_blocks,
    compute_live_map,
    compute_use_defs,
)
from numba_cuda_mlir.numba_cuda.core.errors import ConstantInferenceError

from ._group_planner_support import GroupRewriteError, ir

_VIEW_METHODS = {"view", "reshape", "ravel", "transpose"}
_ARRAY_METADATA = {
    "shape",
    "size",
    "ndim",
    "strides",
    "dtype",
    "items_per_thread",
}
_PayloadPath = tuple[tuple[int, int], ...] | None
_ProgramPoint = tuple[int, int, bool]


def _index_value(index, func_ir):
    if not isinstance(index, ir.Var):
        return index
    try:
        return func_ir.infer_constant(index)
    except ConstantInferenceError:
        # Dynamic slice bounds do not change whether indexing makes a view.
        try:
            definition = func_ir.get_definition(index)
            if (
                isinstance(definition, ir.Expr)
                and definition.op == "call"
                and func_ir.infer_constant(definition.func) is slice
            ):
                return slice(None)
        except (KeyError, ConstantInferenceError):
            pass
    return None


def _is_view_call(value, func_ir):
    try:
        definition = func_ir.get_definition(value.func)
    except KeyError:
        return False
    return (
        isinstance(definition, ir.Expr)
        and definition.op == "getattr"
        and definition.attr in _VIEW_METHODS
    )


def _live_after_statements(blocks, cfg):
    use_defs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, use_defs.usemap, use_defs.defmap)
    result = {}
    for label, block in blocks.items():
        live = set().union(
            *(live_map[next_label] for next_label, _ in cfg.successors(label))
        )
        for statement in reversed(block.body):
            result[statement] = live.copy()
            if isinstance(statement, ir.Assign):
                live.discard(statement.target.name)
                value = statement.value
                if isinstance(value, ir.Var):
                    live.add(value.name)
                elif isinstance(value, ir.Inst):
                    live.update(var.name for var in value.list_vars())
            elif isinstance(statement, ir.Del):
                live.discard(statement.value)
            else:
                live.update(var.name for var in statement.list_vars())
    return result


def _forwarded_paths(
    value,
    references: dict[str, set[_PayloadPath]],
    func_ir,
    tuple_depth: int,
) -> set[_PayloadPath]:
    """Return tuple-index paths to the tracked payload; () is the payload."""
    if isinstance(value, ir.Var):
        return references.get(value.name, set())
    if not isinstance(value, ir.Expr):
        return set()
    if value.op in {"cast", "exhaust_iter"}:
        return references.get(value.value.name, set())
    if value.op == "getattr" and value.attr not in _ARRAY_METADATA:
        return references.get(value.value.name, set())
    if value.op == "call" and _is_view_call(value, func_ir):
        return references.get(value.func.name, set())
    if value.op in {"binop", "inplace_binop"} and value.fn is operator.add:
        paths = references.get(value.lhs.name, set()) | references.get(
            value.rhs.name, set()
        )
        if paths and () not in paths:
            # Concatenated tuples still contain the payload. Retain possible
            # membership without reconstructing the other tuple's shape.
            return {None}
    if value.op == "phi":
        return set().union(
            *(
                references.get(var.name, set())
                for var in value.incoming_values
                if isinstance(var, ir.Var)
            )
        )
    if value.op == "build_tuple":
        paths = set()
        for index, item in enumerate(value.items):
            for path in references.get(item.name, ()):
                # Recursive tuple shapes cannot have a finite Numba type.
                # Widen them instead of growing the analysis indefinitely.
                if path is None or len(path) >= tuple_depth:
                    paths.add(None)
                else:
                    paths.add(((index, len(value.items)), *path))
        return paths
    if value.op in {"getitem", "static_getitem"}:
        index = _index_value(value.index, func_ir)
        paths = set()
        for path in references.get(value.value.name, ()):
            if path is None:
                paths.add(None)
            elif not path:
                if isinstance(index, slice):
                    paths.add(())
            else:
                position, size = path[0]
                if isinstance(index, slice):
                    selected = range(size)[index]
                    if position in selected:
                        paths.add(
                            (
                                (selected.index(position), len(selected)),
                                *path[1:],
                            )
                        )
                elif not isinstance(index, Integral) or index in {
                    position,
                    position - size,
                }:
                    paths.add(path[1:])
        return paths
    return set()


def _reads_payload(statement, references, func_ir):
    """Distinguish forwarding a reference from consuming its stored values."""
    if isinstance(statement, ir.Assign):
        value = statement.value
        if isinstance(value, ir.Var):
            return False
        if isinstance(value, ir.Expr):
            if value.op in {
                "cast",
                "exhaust_iter",
                "phi",
                "build_tuple",
                "getattr",
            }:
                return False
            if value.op in {"getitem", "static_getitem"}:
                paths = references.get(value.value.name, ())
                return None in paths or (
                    () in paths
                    and not isinstance(
                        _index_value(value.index, func_ir), slice
                    )
                )
            if value.op == "call" and _is_view_call(value, func_ir):
                return False
            if value.op == "call":
                try:
                    if func_ir.infer_constant(value.func) is len:
                        return False
                except ConstantInferenceError:
                    pass
            if (
                value.op in {"binop", "inplace_binop"}
                and value.fn is operator.add
            ):
                paths = references.get(value.lhs.name, set()) | references.get(
                    value.rhs.name, set()
                )
                if paths and () not in paths:
                    return False
            return any(var.name in references for var in value.list_vars())
        return False
    if isinstance(statement, ir.Del):
        return False
    return any(var.name in references for var in statement.list_vars())


def validate_load_result_lifetimes(func_ir, calls):
    """Reject a previous result that remains live across the same Load.

    Start after each returning call and trace that generation's references to
    its next execution. Assignment kills the old reference before checking
    liveness, so replacing the result itself is safe. After replacement,
    reject a read through an older reference; merely accessing another tuple
    member or array metadata is safe. Branches merge possible references
    conservatively. Constructors and existing-output Loads are not checked.
    """
    if not calls:
        return
    blocks = func_ir.blocks
    cfg = compute_cfg_from_blocks(blocks)
    loop_blocks = set().union(*(loop.body for loop in cfg.loops().values()))
    if not loop_blocks:
        return
    live_after = _live_after_statements(blocks, cfg)
    tuple_depth = sum(
        isinstance(statement, ir.Assign)
        and isinstance(statement.value, ir.Expr)
        and statement.value.op == "build_tuple"
        for block in blocks.values()
        for statement in block.body
    )
    for label, block in blocks.items():
        if label not in loop_blocks:
            continue
        for index, call in enumerate(block.body):
            if call not in calls:
                continue
            start: _ProgramPoint = (label, index + 1, False)
            incoming: dict[_ProgramPoint, dict[str, set[_PayloadPath]]] = {
                start: {call.target.name: {()}}
            }
            pending: deque[_ProgramPoint] = deque([start])
            while pending:
                point = pending.popleft()
                current_label, offset, overwritten = point
                references = {
                    name: paths.copy()
                    for name, paths in incoming[point].items()
                }
                for statement in blocks[current_label].body[offset:]:
                    if statement is call:
                        references.pop(call.target.name, None)
                        overwritten = True
                        continue
                    if overwritten and _reads_payload(
                        statement, references, func_ir
                    ):
                        raise GroupRewriteError(
                            "cuda.coop.load(items_per_thread=...) cannot "
                            "keep an earlier result live when the same "
                            "call executes again. Numba-CUDA-MLIR reuses "
                            "that call's local storage. Use two explicit "
                            "ThreadData payloads and output-form load "
                            "for values retained across iterations.\n"
                            + call.loc.strformat()
                        )
                    if isinstance(statement, ir.Assign):
                        paths = _forwarded_paths(
                            statement.value, references, func_ir, tuple_depth
                        )
                        references.pop(statement.target.name, None)
                        if paths:
                            references[statement.target.name] = paths
                    elif isinstance(statement, ir.Del):
                        references.pop(statement.value, None)
                    references = {
                        name: paths
                        for name, paths in references.items()
                        if name in live_after[statement]
                    }
                if not references:
                    continue
                for next_label, _ in cfg.successors(current_label):
                    next_point = (next_label, 0, overwritten)
                    previous = incoming.setdefault(next_point, {})
                    changed = False
                    for name, paths in references.items():
                        known = previous.setdefault(name, set())
                        if not paths <= known:
                            known.update(paths)
                            changed = True
                    if changed:
                        pending.append(next_point)
