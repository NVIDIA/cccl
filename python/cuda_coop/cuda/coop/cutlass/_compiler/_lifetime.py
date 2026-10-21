# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate descriptor consumption on the traced kernel's control flow."""

from dataclasses import dataclass
from inspect import currentframe
from typing import Any

from cutlass.base_dsl.common import DSLRuntimeError

from . import _state


@dataclass(frozen=True)
class _LifetimeEvent:
    kernel: Any
    storage: Any
    token: Any
    anchor: Any
    kind: str
    operation: str
    location: str


def _source_location():
    """Keep the originating call site after the Python trace has returned."""
    frame = currentframe()
    try:
        while frame is not None:
            module = frame.f_globals.get("__name__", "")
            if not module.startswith(("cuda.coop", "cutlass")):
                return f"{frame.f_code.co_filename}:{frame.f_lineno}"
            frame = frame.f_back
    finally:
        del frame
    return "unknown source"


def register_creation(storage):
    """Give a kernel-local construction an SSA identity."""
    if _state._active_trace_module_op() is None:
        return None
    from ._storage import _active_cuda_kernel_op, _fresh_i32_placeholder

    session = _state.active_bundle_session()
    token = _fresh_i32_placeholder()
    token = token.ir_value()
    session.add_temp_storage_lifetime_event(
        _LifetimeEvent(
            _active_cuda_kernel_op(),
            storage,
            token,
            token.owner,
            "create",
            "TempStorage",
            _source_location(),
        )
    )
    return token


def register_use(storage, operation, *, release=False):
    """Record calls, including ones with no scratch requirement."""
    from ._storage import _active_cuda_kernel_op, _fresh_i32_placeholder

    if _state._active_trace_module_op() is None:
        return
    if storage._identity_token is None:
        if release:
            raise DSLRuntimeError(
                "TempStorage.release requires a descriptor "
                "constructed inside the kernel."
            )
        return
    session = _state.active_bundle_session()
    marker = _fresh_i32_placeholder()
    session.add_temp_storage_lifetime_event(
        _LifetimeEvent(
            _active_cuda_kernel_op(),
            storage._identity_owner,
            storage._identity_token,
            marker.owner,
            "release" if release else "use",
            operation,
            _source_location(),
        )
    )


def _constant(value):
    owner = value.owner
    if getattr(owner, "name", None) == "arith.constant":
        try:
            return int(owner.attributes["value"].value)
        except (AttributeError, TypeError, ValueError):
            pass
    return None


def _kernel_graph(kernel):
    """Build SCF execution edges and simultaneous SSA argument assignments."""
    edges = {}
    entries = {}

    def node(operation):
        operation = getattr(operation, "operation", operation)
        edges.setdefault(operation, [])
        return operation

    def entry(block):
        # An entry node is distinct from its first operation, including empty
        # blocks. Block arguments are assigned on incoming edges.
        return entries.setdefault(block, (block, "entry"))

    def edge(source, target, destinations=(), values=()):
        if target is not None:
            assert len(destinations) == len(values)
            edges.setdefault(source, []).append(
                (target, tuple(zip(destinations, values)))
            )

    def build_block(block, exits):
        operations = [node(op) for op in block.operations]
        edge(entry(block), operations[0] if operations else None)
        for index, operation in enumerate(operations):
            following = (
                operations[index + 1] if index + 1 < len(operations) else None
            )
            name = operation.name
            if name in {"scf.yield", "scf.condition"}:
                exits(operation)
            elif name == "scf.if":
                condition = _constant(operation.operands[0])
                for branch, region in enumerate(operation.regions):
                    if not region.blocks:
                        if condition is None or not condition:
                            edge(operation, following)
                        continue
                    body = region.blocks[0]
                    if condition is None or bool(condition) == (branch == 0):
                        edge(operation, entry(body))

                    def branch_exit(
                        terminator,
                        *,
                        following=following,
                        results=tuple(operation.results),
                    ):
                        edge(
                            terminator,
                            following,
                            results,
                            tuple(terminator.operands),
                        )

                    build_block(body, branch_exit)
            elif name == "scf.for":
                body = operation.regions[0].blocks[0]
                lower, upper, step = map(_constant, operation.operands[:3])
                count = None
                if None not in (lower, upper, step) and step > 0:
                    count = max(0, (upper - lower + step - 1) // step)
                initial = tuple(operation.operands[3:])
                arguments = tuple(body.arguments[1:])
                results = tuple(operation.results)
                if count is None or count:
                    edge(operation, entry(body), arguments, initial)
                if count is None or count == 0:
                    edge(operation, following, results, initial)

                def for_exit(
                    terminator,
                    *,
                    body=body,
                    count=count,
                    arguments=arguments,
                    results=results,
                    following=following,
                ):
                    values = tuple(terminator.operands)
                    edge(terminator, following, results, values)
                    if count is None or count > 1:
                        edge(terminator, entry(body), arguments, values)

                build_block(body, for_exit)
            elif name == "scf.while":
                before = operation.regions[0].blocks[0]
                after = operation.regions[1].blocks[0]
                edge(
                    operation,
                    entry(before),
                    tuple(before.arguments),
                    tuple(operation.operands),
                )

                def condition_exit(
                    terminator,
                    *,
                    after=after,
                    following=following,
                    results=tuple(operation.results),
                ):
                    condition = _constant(terminator.operands[0])
                    values = tuple(terminator.operands[1:])
                    if condition is None or condition:
                        edge(
                            terminator,
                            entry(after),
                            tuple(after.arguments),
                            values,
                        )
                    if condition is None or not condition:
                        edge(terminator, following, results, values)

                build_block(before, condition_exit)
                build_block(
                    after,
                    lambda terminator, before=before: edge(
                        terminator,
                        entry(before),
                        tuple(before.arguments),
                        tuple(terminator.operands),
                    ),
                )
            elif name == "scf.execute_region":
                edge(operation, entry(operation.regions[0].blocks[0]))
                for body in operation.regions[0].blocks:

                    def region_exit(
                        terminator,
                        *,
                        following=following,
                        results=tuple(operation.results),
                    ):
                        edge(
                            terminator,
                            following,
                            results,
                            tuple(terminator.operands),
                        )

                    build_block(body, region_exit)
            elif name in {"cf.br", "cf.cond_br"}:
                values = list(operation.operands)
                condition = None
                if name == "cf.cond_br":
                    condition = _constant(values.pop(0))
                for branch, target in enumerate(operation.successors):
                    count = len(target.arguments)
                    if condition is None or bool(condition) == (branch == 0):
                        edge(
                            operation,
                            entry(target),
                            tuple(target.arguments),
                            values[:count],
                        )
                    del values[:count]
            elif operation.regions or operation.successors:
                raise DSLRuntimeError(
                    f"TempStorage.release cannot analyze control flow {name} "
                    f"({operation.location})."
                )
            else:
                edge(operation, following)

    blocks = kernel.regions[0].blocks
    for block in blocks:
        build_block(block, lambda terminator: None)
    return entry(blocks[0]), edges


def validate_temp_storage_lifetimes(session):
    """Reject any reachable use after release, preserving alias consumption."""
    events = session.temp_storage_lifetime_event_list()
    if not any(event.kind == "release" for event in events):
        return
    kernels = {event.kernel for event in events}
    for kernel in kernels:
        local = [event for event in events if event.kernel == kernel]
        if not any(event.kind == "release" for event in local):
            continue
        indexed = {
            event.anchor: (index, event) for index, event in enumerate(local)
        }
        origins = {
            id(event.storage): index
            for index, event in enumerate(local)
            if event.kind == "create"
        }
        first, edges = _kernel_graph(kernel)
        incoming = {first: {}}
        pending = [first]
        seen_uses = {}
        while pending:
            current = pending.pop()
            state = {
                value: set(owners)
                for value, owners in incoming[current].items()
            }
            record = indexed.get(current)
            if record:
                index, event = record
                if event.kind == "create":
                    state[event.token] = {(index, None)}
                else:
                    owners = state.get(event.token, set())
                    seen_uses[index] = owners
                    consumed = [
                        release for _, release in owners if release is not None
                    ]
                    if consumed:
                        previous = local[consumed[0]]
                        raise DSLRuntimeError(
                            "TempStorage cannot be used by "
                            f"{event.operation} after "
                            "release() on a reachable path: "
                            f"released at {previous.location}; "
                            f"used at {event.location}."
                        )
                    if owners and {origin for origin, _ in owners} != {
                        origins.get(id(event.storage))
                    }:
                        raise DSLRuntimeError(
                            "TempStorage cannot select different descriptor "
                            "instances through runtime control flow "
                            f"({event.location})."
                        )
                    if event.kind == "release":
                        selected = {origin for origin, _ in owners}
                        state = {
                            value: {
                                (
                                    origin,
                                    index if origin in selected else release,
                                )
                                for origin, release in values
                            }
                            for value, values in state.items()
                        }
            for target, assignments in edges.get(current, ()):
                transferred = dict(state)
                for destination, value in assignments:
                    transferred[destination] = set(state.get(value, ()))
                previous = incoming.get(target)
                merged = (
                    {}
                    if previous is None
                    else {
                        value: set(owners) for value, owners in previous.items()
                    }
                )
                for value, owners in transferred.items():
                    merged.setdefault(value, set()).update(owners)
                if previous is None or merged != previous:
                    incoming[target] = merged
                    pending.append(target)
        for index, owners in seen_uses.items():
            if not owners:
                raise DSLRuntimeError(
                    "TempStorage use has no local descriptor instance "
                    f"({local[index].location})."
                )


def erase_temp_storage_lifetime_markers(session):
    """Erase analysis anchors and unused identity constants."""
    for event in session.temp_storage_lifetime_event_list():
        if event.kind != "create" or not list(event.token.uses):
            event.anchor.erase()
