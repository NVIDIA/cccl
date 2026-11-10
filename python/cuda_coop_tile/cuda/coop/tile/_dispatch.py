# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Choose Python-side defaults and topologies for block-adapter checks.

These heuristics run while cuTile lowers a public call. They do not inspect
the enclosing kernel's final worker count. Except for Exchange's semantic
width, the foreign declaration leaves physical topology to the native
provider and compiler. The dispatch trace describes this earlier decision.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from cuda.tile import _datatype as datatype
from cuda.tile._exception import TileTypeError

from ._block.exchange import _SUPPORTED_DTYPES as _EXCHANGE_DTYPES

_SCAN_WARP_ALG = "warp_scans"
_SCAN_BLOCK_ALG = "raking"
_REDUCE_WARP_ALG = "warp_reductions"
_THREAD_CANDIDATES = (256, 128, 64, 32)
_ONE_ITEM_THREAD_CANDIDATES = (128, 64, 32)


@dataclass(frozen=True)
class CoopDispatchDecision:
    """Record a Python validation choice and its diagnostic explanation.

    ``block_threads`` gives the block adapter a concrete item count for its
    checks. Ordinary foreign declarations retain the logical extent, so the
    native provider and compiler still select the implementation width.
    Exchange carries this width into the declaration because it defines the
    permutation. ``backend_family`` describes a Python heuristic; it does not
    establish the selected native implementation's execution scope.
    """

    primitive: str
    backend_family: str
    block_threads: int
    algorithm: str | None
    rationale: str
    tile_numel: int
    dtype: datatype.DType


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _trace_dispatch(decision: CoopDispatchDecision) -> None:
    if not _env_truthy("CUDA_COOP_TILE_DISPATCH_TRACE"):
        return
    msg = (
        "[cuda.coop.tile.dispatch] "
        f"primitive={decision.primitive} "
        f"backend={decision.backend_family} "
        f"threads={decision.block_threads} "
        f"algorithm={decision.algorithm} "
        f"numel={decision.tile_numel} "
        f"dtype={decision.dtype} "
        f"reason={decision.rationale}"
    )
    print(msg, file=sys.stderr)


def _select_block_threads(
    *,
    primitive: str,
    numel: int,
    preferred: int,
) -> int:
    """Choose the first candidate width that divides the logical tile.

    Try ``preferred`` before the remaining fixed candidates. The result lets
    block adapters validate a whole number of items per thread; it is not a
    measurement of the best finished-kernel configuration. Reject nonpositive
    extents and extents with no supported divisor.
    """

    if not isinstance(numel, int):
        raise TileTypeError(
            f"coop.{primitive} expects an integer tile size, got "
            f"{type(numel).__name__}"
        )
    if numel <= 0:
        raise TileTypeError(
            f"coop.{primitive} expects tile size > 0, got {numel}"
        )
    candidates = [preferred]
    candidates.extend(t for t in _THREAD_CANDIDATES if t != preferred)
    for threads in candidates:
        if numel % threads == 0:
            return threads
    raise TileTypeError(
        f"coop.{primitive} does not support tile size {numel}; "
        f"tile size must be divisible by one of "
        f"{tuple(dict.fromkeys(candidates))}"
    )


def _select_one_item_block_threads(*, primitive: str, numel: int) -> int:
    """Prefer one item per thread for supported small sorting tiles.

    For larger extents, use the ordinary 128-thread preference and divisor
    search. The native provider later checks its own candidate constraints.
    """

    if not isinstance(numel, int):
        raise TileTypeError(
            f"coop.{primitive} expects an integer tile size, got "
            f"{type(numel).__name__}"
        )
    if numel <= 0:
        raise TileTypeError(
            f"coop.{primitive} expects tile size > 0, got {numel}"
        )
    if numel in _ONE_ITEM_THREAD_CANDIDATES:
        return numel
    return _select_block_threads(
        primitive=primitive,
        numel=numel,
        preferred=128,
    )


def _histogram_thread_candidates(preferred: int) -> tuple[int, ...]:
    candidates = [preferred]
    candidates.extend(t for t in _THREAD_CANDIDATES if t != preferred)
    return tuple(dict.fromkeys(candidates))


def _select_histogram_block_threads(
    *,
    primitive: str,
    numel: int,
    bins: int,
    preferred: int,
) -> int:
    """Find a width that divides both the input and histogram output.

    Each candidate must give integral samples and bins per thread and use
    whole warps, as required by the native provider.
    """

    if not isinstance(numel, int):
        raise TileTypeError(
            f"coop.{primitive} expects an integer tile size, got "
            f"{type(numel).__name__}"
        )
    if numel <= 0:
        raise TileTypeError(
            f"coop.{primitive} expects tile size > 0, got {numel}"
        )

    candidates = _histogram_thread_candidates(preferred)
    for threads in candidates:
        if numel % threads == 0 and bins % threads == 0:
            return threads
    raise TileTypeError(
        f"coop.{primitive} requires a block-thread count that divides both "
        f"tile size ({numel}) and bins ({bins}); tried {candidates}"
    )


def _forced_backend() -> str | None:
    forced = os.getenv("CUDA_COOP_TILE_DISPATCH_FORCE_BACKEND")
    if forced is None:
        return None
    forced = forced.strip().lower()
    if forced in ("", "auto"):
        return None
    if forced not in ("warp_like", "cta_block"):
        raise TileTypeError(
            "CUDA_COOP_TILE_DISPATCH_FORCE_BACKEND must be one of: "
            "warp_like, cta_block, auto"
        )
    return forced


def _default_backend_for_numel(numel: int) -> tuple[str, str]:
    if numel <= 64 and numel % 32 == 0:
        return (
            "warp_like",
            "small tile heuristic (numel <= 64 and divisible by 32)",
        )
    return "cta_block", "default block heuristic"


def _apply_forced_backend(
    backend: str,
    rationale: str,
) -> tuple[str, str]:
    forced = _forced_backend()
    if forced is None:
        return backend, rationale
    return forced, f"forced via CUDA_COOP_TILE_DISPATCH_FORCE_BACKEND={forced}"


def _preferred_threads(backend: str, numel: int) -> int:
    if backend == "warp_like" and numel <= 64 and numel in _THREAD_CANDIDATES:
        return numel
    return 128


def _dispatch_scan_like(
    *,
    primitive: str,
    numel: int,
    dtype: datatype.DType,
    requested_algorithm: str | None,
) -> CoopDispatchDecision:
    """Normalize the Scan algorithm and record a validation topology.

    An explicit algorithm sets the initial heuristic family. The environment
    override can change that family, but does not replace an explicit
    algorithm. Omitted algorithms match the C++ scan policy's raking
    default, independently of the physical validation width.
    """

    if requested_algorithm == _SCAN_WARP_ALG:
        backend = "warp_like"
        rationale = "algorithm requests warp scan backend"
    elif requested_algorithm in ("raking", "raking_memoize"):
        backend = "cta_block"
        rationale = "algorithm requests block scan backend"
    else:
        backend, rationale = _default_backend_for_numel(numel)

    backend, rationale = _apply_forced_backend(backend, rationale)
    preferred_threads = _preferred_threads(backend, numel)
    block_threads = _select_block_threads(
        primitive=primitive,
        numel=numel,
        preferred=preferred_threads,
    )

    algorithm = requested_algorithm
    if algorithm is None:
        algorithm = _SCAN_BLOCK_ALG

    decision = CoopDispatchDecision(
        primitive=primitive,
        backend_family=backend,
        block_threads=block_threads,
        algorithm=algorithm,
        rationale=rationale,
        tile_numel=numel,
        dtype=dtype,
    )
    _trace_dispatch(decision)
    return decision


def _dispatch_reduce_like(
    *,
    primitive: str,
    numel: int,
    dtype: datatype.DType,
    requested_algorithm: str | None,
) -> CoopDispatchDecision:
    """Normalize the Reduce algorithm before block-adapter validation.

    Follow the same explicit-algorithm and environment precedence as Scan,
    using Reduce's own CUB algorithm names. The native provider receives the
    algorithm and logical extent, not this heuristic's physical width.
    """

    if requested_algorithm == _REDUCE_WARP_ALG:
        backend = "warp_like"
        rationale = "algorithm requests warp reduction backend"
    elif requested_algorithm in ("raking", "raking_commutative_only"):
        backend = "cta_block"
        rationale = "algorithm requests block reduction backend"
    else:
        backend, rationale = _default_backend_for_numel(numel)

    backend, rationale = _apply_forced_backend(backend, rationale)
    preferred_threads = _preferred_threads(backend, numel)
    # The reduction provider supports valid inputs smaller than a warp.
    # This width validates the adapter; final worker selection stays native.
    if isinstance(numel, int) and 0 < numel < 32:
        if 32 % numel:
            raise TileTypeError(
                f"coop.{primitive} row extent {numel} must divide 32 "
                "when smaller than a warp"
            )
        block_threads = 32
    else:
        block_threads = _select_block_threads(
            primitive=primitive,
            numel=numel,
            preferred=preferred_threads,
        )

    algorithm = requested_algorithm
    if algorithm is None:
        algorithm = _REDUCE_WARP_ALG

    decision = CoopDispatchDecision(
        primitive=primitive,
        backend_family=backend,
        block_threads=block_threads,
        algorithm=algorithm,
        rationale=rationale,
        tile_numel=numel,
        dtype=dtype,
    )
    _trace_dispatch(decision)
    return decision


def dispatch_exclusive_sum(
    *,
    numel: int,
    dtype: datatype.DType,
    requested_algorithm: str | None,
) -> CoopDispatchDecision:
    return _dispatch_scan_like(
        primitive="exclusive_sum",
        numel=numel,
        dtype=dtype,
        requested_algorithm=requested_algorithm,
    )


def dispatch_inclusive_sum(
    *,
    numel: int,
    dtype: datatype.DType,
    requested_algorithm: str | None,
) -> CoopDispatchDecision:
    return _dispatch_scan_like(
        primitive="inclusive_sum",
        numel=numel,
        dtype=dtype,
        requested_algorithm=requested_algorithm,
    )


def dispatch_exclusive_scan(
    *,
    numel: int,
    dtype: datatype.DType,
    requested_algorithm: str | None,
) -> CoopDispatchDecision:
    return _dispatch_scan_like(
        primitive="exclusive_scan",
        numel=numel,
        dtype=dtype,
        requested_algorithm=requested_algorithm,
    )


def dispatch_inclusive_scan(
    *,
    numel: int,
    dtype: datatype.DType,
    requested_algorithm: str | None,
) -> CoopDispatchDecision:
    return _dispatch_scan_like(
        primitive="inclusive_scan",
        numel=numel,
        dtype=dtype,
        requested_algorithm=requested_algorithm,
    )


def dispatch_sum(
    *,
    numel: int,
    dtype: datatype.DType,
    requested_algorithm: str | None,
) -> CoopDispatchDecision:
    return _dispatch_reduce_like(
        primitive="sum",
        numel=numel,
        dtype=dtype,
        requested_algorithm=requested_algorithm,
    )


def dispatch_reduce(
    *,
    numel: int,
    dtype: datatype.DType,
    requested_algorithm: str | None,
) -> CoopDispatchDecision:
    return _dispatch_reduce_like(
        primitive="reduce",
        numel=numel,
        dtype=dtype,
        requested_algorithm=requested_algorithm,
    )


def dispatch_histogram_256(
    *,
    numel: int,
    dtype: datatype.DType,
    bins: int,
) -> CoopDispatchDecision:
    return dispatch_histogram(
        numel=numel,
        dtype=dtype,
        bins=bins,
        primitive="histogram_256",
    )


def dispatch_histogram(
    *,
    numel: int,
    dtype: datatype.DType,
    bins: int,
    primitive: str = "histogram",
) -> CoopDispatchDecision:
    if not isinstance(bins, int):
        raise TileTypeError(f"coop.{primitive} bins must be an int")
    if bins <= 0:
        raise TileTypeError(f"coop.{primitive} bins must be > 0")
    if bins > 256:
        raise TileTypeError(
            f"coop.{primitive} bins must be <= 256 for uint8 inputs"
        )

    backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)
    preferred_threads = _preferred_threads(backend, numel)
    block_threads = _select_histogram_block_threads(
        primitive=primitive,
        numel=numel,
        bins=bins,
        preferred=preferred_threads,
    )

    decision = CoopDispatchDecision(
        primitive=primitive,
        backend_family=backend,
        block_threads=block_threads,
        algorithm=None,
        rationale=rationale,
        tile_numel=numel,
        dtype=dtype,
    )
    _trace_dispatch(decision)
    return decision


def dispatch_merge_sort_keys(
    *,
    numel: int,
    dtype: datatype.DType,
    descending: bool,
    valid_items: int | None,
) -> CoopDispatchDecision:
    del descending
    del valid_items
    backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)
    block_threads = _select_one_item_block_threads(
        primitive="merge_sort_keys",
        numel=numel,
    )
    decision = CoopDispatchDecision(
        primitive="merge_sort_keys",
        backend_family=backend,
        block_threads=block_threads,
        algorithm=None,
        rationale=rationale,
        tile_numel=numel,
        dtype=dtype,
    )
    _trace_dispatch(decision)
    return decision


def dispatch_merge_sort_pairs(
    *,
    numel: int,
    key_dtype: datatype.DType,
    value_dtype: datatype.DType,
    descending: bool,
    valid_items: int | None,
) -> CoopDispatchDecision:
    del value_dtype
    del descending
    del valid_items
    backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)
    block_threads = _select_one_item_block_threads(
        primitive="merge_sort_pairs",
        numel=numel,
    )
    decision = CoopDispatchDecision(
        primitive="merge_sort_pairs",
        backend_family=backend,
        block_threads=block_threads,
        algorithm=None,
        rationale=rationale,
        tile_numel=numel,
        dtype=key_dtype,
    )
    _trace_dispatch(decision)
    return decision


def dispatch_radix_sort_keys(
    *,
    numel: int,
    dtype: datatype.DType,
    inner_scan_algorithm: str | None,
) -> CoopDispatchDecision:
    if inner_scan_algorithm == "warp_scans":
        backend = "warp_like"
        rationale = "inner_scan_algorithm requests warp scan backend"
    elif inner_scan_algorithm in ("raking", "raking_memoize"):
        backend = "cta_block"
        rationale = "inner_scan_algorithm requests block scan backend"
    else:
        backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)
    block_threads = _select_one_item_block_threads(
        primitive="radix_sort_keys",
        numel=numel,
    )
    decision = CoopDispatchDecision(
        primitive="radix_sort_keys",
        backend_family=backend,
        block_threads=block_threads,
        algorithm=inner_scan_algorithm,
        rationale=rationale,
        tile_numel=numel,
        dtype=dtype,
    )
    _trace_dispatch(decision)
    return decision


def dispatch_radix_sort_pairs(
    *,
    numel: int,
    key_dtype: datatype.DType,
    value_dtype: datatype.DType,
    inner_scan_algorithm: str | None,
) -> CoopDispatchDecision:
    del value_dtype
    if inner_scan_algorithm == "warp_scans":
        backend = "warp_like"
        rationale = "inner_scan_algorithm requests warp scan backend"
    elif inner_scan_algorithm in ("raking", "raking_memoize"):
        backend = "cta_block"
        rationale = "inner_scan_algorithm requests block scan backend"
    else:
        backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)
    block_threads = _select_one_item_block_threads(
        primitive="radix_sort_pairs",
        numel=numel,
    )
    decision = CoopDispatchDecision(
        primitive="radix_sort_pairs",
        backend_family=backend,
        block_threads=block_threads,
        algorithm=inner_scan_algorithm,
        rationale=rationale,
        tile_numel=numel,
        dtype=key_dtype,
    )
    _trace_dispatch(decision)
    return decision


def dispatch_topk_keys(
    *,
    numel: int,
    dtype: datatype.DType,
) -> CoopDispatchDecision:
    backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)
    block_threads = _select_one_item_block_threads(
        primitive="topk_keys",
        numel=numel,
    )
    decision = CoopDispatchDecision(
        primitive="topk_keys",
        backend_family=backend,
        block_threads=block_threads,
        algorithm=None,
        rationale=rationale,
        tile_numel=numel,
        dtype=dtype,
    )
    _trace_dispatch(decision)
    return decision


def dispatch_topk_pairs(
    *,
    numel: int,
    key_dtype: datatype.DType,
    value_dtype: datatype.DType,
) -> CoopDispatchDecision:
    del value_dtype
    backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)
    block_threads = _select_one_item_block_threads(
        primitive="topk_pairs",
        numel=numel,
    )
    decision = CoopDispatchDecision(
        primitive="topk_pairs",
        backend_family=backend,
        block_threads=block_threads,
        algorithm=None,
        rationale=rationale,
        tile_numel=numel,
        dtype=key_dtype,
    )
    _trace_dispatch(decision)
    return decision


def dispatch_discontinuity(
    *,
    primitive: str,
    numel: int,
    dtype: datatype.DType,
) -> CoopDispatchDecision:
    backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)
    preferred_threads = _preferred_threads(backend, numel)
    block_threads = _select_block_threads(
        primitive=primitive,
        numel=numel,
        preferred=preferred_threads,
    )
    decision = CoopDispatchDecision(
        primitive=primitive,
        backend_family=backend,
        block_threads=block_threads,
        algorithm=None,
        rationale=rationale,
        tile_numel=numel,
        dtype=dtype,
    )
    _trace_dispatch(decision)
    return decision


def dispatch_adjacent_difference_subtract_left(
    *,
    numel: int,
    dtype: datatype.DType,
) -> CoopDispatchDecision:
    backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)
    preferred_threads = _preferred_threads(backend, numel)
    block_threads = _select_block_threads(
        primitive="adjacent_difference_subtract_left",
        numel=numel,
        preferred=preferred_threads,
    )
    decision = CoopDispatchDecision(
        primitive="adjacent_difference_subtract_left",
        backend_family=backend,
        block_threads=block_threads,
        algorithm=None,
        rationale=rationale,
        tile_numel=numel,
        dtype=dtype,
    )
    _trace_dispatch(decision)
    return decision


def dispatch_adjacent_difference_subtract_right(
    *,
    numel: int,
    dtype: datatype.DType,
) -> CoopDispatchDecision:
    backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)
    preferred_threads = _preferred_threads(backend, numel)
    block_threads = _select_block_threads(
        primitive="adjacent_difference_subtract_right",
        numel=numel,
        preferred=preferred_threads,
    )
    decision = CoopDispatchDecision(
        primitive="adjacent_difference_subtract_right",
        backend_family=backend,
        block_threads=block_threads,
        algorithm=None,
        rationale=rationale,
        tile_numel=numel,
        dtype=dtype,
    )
    _trace_dispatch(decision)
    return decision


def _validate_exchange_topology(
    primitive: str, numel: int, block_threads: int
) -> None:
    """Check the width that defines an Exchange permutation.

    Unlike a tuning preference, this width is part of the operation's meaning.
    Require complete warps and an integral number of elements per member.
    """

    if (
        isinstance(block_threads, bool)
        or not isinstance(block_threads, int)
        or not 32 <= block_threads <= 1024
        or block_threads % 32 != 0
    ):
        raise TileTypeError(
            f"coop.{primitive} block_threads must be a compile-time integer "
            "multiple of 32 in [32, 1024]"
        )
    if numel <= 0 or numel % block_threads != 0:
        raise TileTypeError(
            f"coop.{primitive} tile extent must be positive and "
            f"divisible by block_threads"
        )


def dispatch_exchange_striped_to_blocked(
    *,
    numel: int,
    dtype: datatype.DType,
    block_threads: int,
) -> CoopDispatchDecision:
    backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)
    _validate_exchange_topology(
        "exchange_striped_to_blocked", numel, block_threads
    )
    if dtype not in _EXCHANGE_DTYPES:
        raise TileTypeError(
            f"coop.exchange_striped_to_blocked does not support dtype {dtype}"
        )
    decision = CoopDispatchDecision(
        primitive="exchange_striped_to_blocked",
        backend_family=backend,
        block_threads=block_threads,
        algorithm=None,
        rationale=rationale,
        tile_numel=numel,
        dtype=dtype,
    )
    _trace_dispatch(decision)
    return decision


def dispatch_exchange_blocked_to_striped(
    *,
    numel: int,
    dtype: datatype.DType,
    block_threads: int,
) -> CoopDispatchDecision:
    backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)
    _validate_exchange_topology(
        "exchange_blocked_to_striped", numel, block_threads
    )
    if dtype not in _EXCHANGE_DTYPES:
        raise TileTypeError(
            f"coop.exchange_blocked_to_striped does not support dtype {dtype}"
        )
    decision = CoopDispatchDecision(
        primitive="exchange_blocked_to_striped",
        backend_family=backend,
        block_threads=block_threads,
        algorithm=None,
        rationale=rationale,
        tile_numel=numel,
        dtype=dtype,
    )
    _trace_dispatch(decision)
    return decision


def dispatch_shuffle(
    *,
    primitive: str,
    numel: int,
    dtype: datatype.DType,
    mode: str,
) -> CoopDispatchDecision:
    """Choose a validation width for the requested Shuffle overload.

    Offset and Rotate require one item per thread and therefore an extent
    from the fixed thread candidates. Up and Down allow several items per
    thread. The returned decision also records the mode for diagnostics.
    """

    backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)

    if mode in ("offset", "rotate"):
        if not isinstance(numel, int):
            raise TileTypeError(
                f"coop.{primitive} expects an integer tile size, "
                f"got {type(numel).__name__}"
            )
        if numel <= 0:
            raise TileTypeError(
                f"coop.{primitive} expects tile size > 0, got {numel}"
            )
        if numel % 32 or numel > 1024:
            raise TileTypeError(
                f"coop.{primitive} requires one item per thread; tile size "
                "must be a multiple of 32 in [32, 1024]"
            )
        block_threads = numel
    else:
        preferred_threads = _preferred_threads(backend, numel)
        block_threads = _select_block_threads(
            primitive=primitive,
            numel=numel,
            preferred=preferred_threads,
        )

    decision = CoopDispatchDecision(
        primitive=primitive,
        backend_family=backend,
        block_threads=block_threads,
        algorithm=mode,
        rationale=rationale,
        tile_numel=numel,
        dtype=dtype,
    )
    _trace_dispatch(decision)
    return decision


def dispatch_radix_rank(
    *,
    numel: int,
    dtype: datatype.DType,
) -> CoopDispatchDecision:
    backend, rationale = _default_backend_for_numel(numel)
    backend, rationale = _apply_forced_backend(backend, rationale)
    preferred_threads = _preferred_threads(backend, numel)
    block_threads = _select_block_threads(
        primitive="radix_rank_keys",
        numel=numel,
        preferred=preferred_threads,
    )
    decision = CoopDispatchDecision(
        primitive="radix_rank_keys",
        backend_family=backend,
        block_threads=block_threads,
        algorithm=None,
        rationale=rationale,
        tile_numel=numel,
        dtype=dtype,
    )
    _trace_dispatch(decision)
    return decision
