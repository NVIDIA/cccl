# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Choose a provider specialization for each site, once the block is known.

Specialization planning is the second stage of ``CoopCallResolution``. From
the checked sites and the launch's block it:

* completes each distinct request with coop's ``plan_group_primitive``,
  memoized per request and block, and specializes its provider at the
  kernel's target, with one compile context for all of them;
* decides ``launch_bounds`` from the block's thread count;
* decides which sites take scratch, from each provider's storage ABI, and
  places a barrier after every one of them, as coop's planner does.

Specializing generates provider source and nothing more: this stage does
not call NVRTC, so scratch use is decided before any size is known. It does
not change the IR or the compile state; the driver applies its decisions.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from ._errors import TypedPipelineError
from ._sites import Sites
from ._templates import LoadStoreRequest, OperationRequest


@dataclass(frozen=True)
class LaunchBounds:
    """The kernel's ``launch_bounds``; ``rebind`` when the stage chose it."""

    value: int | None
    rebind: bool = False


@dataclass(frozen=True)
class Specialization:
    """What the kernel's sites specialized to, one entry per site.

    ``algorithms`` holds each distinct provider once, in first-use order;
    ``site_algorithms`` repeats them per site.
    """

    block: tuple[int, int, int]
    algorithms: tuple
    site_algorithms: tuple
    scratch: tuple[bool, ...]
    barrier_after: tuple[bool, ...]
    launch_bounds: LaunchBounds
    lowering_plans: tuple = ()


# Request completion ----------------------------------------------------------

_COMPLETIONS: dict[tuple, tuple[Any, dict]] = {}
_LOWERING_PLANS: dict[tuple, Any] = {}


def lowering_plan(request: Any, block: tuple) -> Any:
    """Resolve the shared operation, topology and storage contract once."""

    from ..._core import plan_group_primitive
    from ..._core.launch import LaunchFacts
    from .._compiler._storage_layout import apply_caller_storage

    key = (request.key, block)
    if key not in _LOWERING_PLANS:
        try:
            plan = plan_group_primitive(
                request.call, LaunchFacts(exact_block_dim=block)
            ).require_supported()
            options = dict(getattr(request, "storage_options", ()))
            if options:
                plan = apply_caller_storage(
                    plan,
                    (
                        options.get("storage_size_in_bytes"),
                        options.get("storage_alignment"),
                        options.get("storage_auto_sync", False),
                        options.get("storage_sharing", "shared"),
                    ),
                    operation=request.kind,
                )
            _LOWERING_PLANS[key] = plan
        except (NotImplementedError, TypeError, ValueError) as exc:
            raise TypedPipelineError(f"coop.{request.kind}: {exc}") from None
    return _LOWERING_PLANS[key]


def _complete(request: OperationRequest, block: tuple) -> tuple[Any, dict]:
    """Return the provider factory and its arguments for ``request``.

    ``plan_group_primitive`` hashes the request's semantic key, which is
    costly, so its result is memoized per request and block.
    """

    key = (request.key, block)
    completed = _COMPLETIONS.get(key)
    if completed is not None:
        return completed
    resolved = lowering_plan(request, block)
    complete_request = getattr(request, "complete", None)
    if complete_request is not None:
        completed = complete_request(block, resolved)
        _COMPLETIONS[key] = completed
        return completed

    from ._load_store import complete

    assert isinstance(request, LoadStoreRequest)
    completed = complete(request, block, resolved)
    _COMPLETIONS[key] = completed
    return completed


@contextlib.contextmanager
def _one_compile_context(context: Any) -> Iterator[None]:
    """Make every provider built inside the block use ``context``.

    Coop's load/store factories resolve their own compile context and take
    none from the caller. Compiles run under Numba's compiler lock,
    so replacing the resolver for the duration of a compile is not visible
    to another compile.
    """

    from .._compiler import _nvrtc

    resolve = _nvrtc.resolve_compile_context
    _nvrtc.resolve_compile_context = lambda: context
    try:
        yield
    finally:
        _nvrtc.resolve_compile_context = resolve


def _instantiate(
    requests: Sequence[OperationRequest], block: tuple, target, context
) -> list:
    from .._types import collect_specializations, kernel_compile_target

    with (
        kernel_compile_target(target),
        _one_compile_context(context),
        collect_specializations() as collected,
    ):
        for request in requests:
            factory, kwargs = _complete(request, block)
            factory(**kwargs)
    if len(collected) != len(requests):
        raise AssertionError("a provider factory did not specialize once")
    return collected


# Decisions -------------------------------------------------------------------


def uses_scratch(algorithm: Any) -> bool:
    """Whether ``algorithm``'s provider takes the scratch pointer."""

    from .._compiler._operations import StorageABI

    return algorithm.storage_abi is StorageABI.LEADING_POINTER


def plan_launch_bounds(block: tuple, targetoptions: dict) -> LaunchBounds:
    threads = block[0] * block[1] * block[2]
    bounds = targetoptions.get("launch_bounds")
    if bounds is not None:
        maximum = bounds[0] if isinstance(bounds, tuple) else bounds
        if threads > maximum:
            raise TypedPipelineError(
                f"the launch's block {block} has {threads} threads, "
                f"more than launch_bounds={bounds!r}"
            )
        return LaunchBounds(maximum)
    if targetoptions.get("max_registers") is None:
        return LaunchBounds(threads, rebind=True)
    return LaunchBounds(None)


def plan(
    algorithms: Sequence[Any],
    site_algorithms: Sequence[Any],
    block: tuple,
    targetoptions: dict,
    *,
    lowering_plans: Sequence[Any] = (),
) -> Specialization:
    """Decide scratch, barriers and ``launch_bounds`` for specialized sites.

    A barrier follows every call that uses scratch.
    """

    launch_bounds = plan_launch_bounds(block, targetoptions)
    scratch = tuple(uses_scratch(a) for a in site_algorithms)
    return Specialization(
        block=block,
        algorithms=tuple(algorithms),
        site_algorithms=tuple(site_algorithms),
        scratch=scratch,
        barrier_after=scratch,
        launch_bounds=launch_bounds,
        lowering_plans=tuple(lowering_plans),
    )


def specialize(
    sites: Sites,
    block: tuple,
    *,
    target: Any,
    context: Any,
    targetoptions: dict,
) -> Specialization:
    """Specialize every site's provider for ``block`` and plan around them."""

    unique = list(dict.fromkeys(s.request for s in sites.sites))
    algorithms = _instantiate(unique, block, target, context)
    algorithm_for = dict(zip(unique, algorithms))
    return plan(
        algorithms,
        [algorithm_for[s.request] for s in sites.sites],
        block,
        targetoptions,
        lowering_plans=[lowering_plan(s.request, block) for s in sites.sites],
    )


__all__ = [
    "LaunchBounds",
    "Specialization",
    "lowering_plan",
    "plan",
    "plan_launch_bounds",
    "specialize",
    "uses_scratch",
]
