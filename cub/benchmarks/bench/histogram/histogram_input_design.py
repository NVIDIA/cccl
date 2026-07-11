#!/usr/bin/env python3
"""Bit-exact Python mirror of the CUB histogram input-shape generators.

SOURCE OF TRUTH: cub/benchmarks/bench/histogram/histogram_inputs.cuh.
This module is a faithful host-side port of that header, NOT an independent
design sketch. Every shape, knob, default, and the RNG/scatter math below mirror
the C++ exactly, so the bin indices produced here equal what the benchmark
produces on device for the same (shape, n, num_bins, seed). Keep them in sync:
if the .cuh changes, update this file.

The C++ replaced the legacy bitwise-AND "entropy" knob with a tunable
`InputShape` axis whose values are `name[:knob]`. There is ONE `concentrated`
shape spanning uniform<->constant via its entropy knob (no separate
uniform/constant/spike names), plus multi-hot (powerlaw) and
cache-adversarial (hash_synonym/stale_resident) and ordering
(temporal_phases/strided_sweep/sawtooth) shapes.

Mechanism mirrored from the header:
  * Every shape decides a per-element BIN index in [0, num_bins); a bin->value
    mapper then emits a SampleT in the bin's interval (EVEN: midpoint; RANGE:
    level-interval midpoint). CUB re-derives the bin, so the in-bench verifier
    validates the mapping.
  * i.i.d. distribution shapes: build a host pmf -> inclusive CDF -> per element
    draw u01_from_hash(element_key(i, seed)) and upper_bound_cdf into a bin.
  * ordering shapes (stale_resident/temporal_phases/strided_sweep/sawtooth/poison):
    positional functors, order is intrinsic.
  * the concentrated uniform endpoint (knob>=1.0) is an exact equal-count tiling
    in a deterministic shuffled order.
  * hot bins are scattered off zero via scatter_bin (a fixed coprime
    permutation seeded by seed%num_bins), so the mode is never bin 0.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants — mirror histogram_inputs.cuh exactly.
# ---------------------------------------------------------------------------
K_HASH_SYNONYM_COUNT = 32  # kHashSynonymCount
# Prime larger than every configured bin count. On power-of-two axes it is -1.
# Bounded working sets use a separate upper-window mapping so this constant can
# never move their support.
K_SCATTER_PRIME = 2147483647  # kScatterPrime (2^31 - 1)
K_CHANNEL_SEED_STRIDE = 0xD1B54A32D192ED03

DEFAULT_CONCENTRATED_ENTROPY = 0.5  # bare "concentrated"
DEFAULT_POWERLAW_ENTROPY = 0.5  # bare "powerlaw"
DEFAULT_HASH_SYNONYM_HOTSHARE = 0.9  # bare "hash_synonym"
DEFAULT_TEMPORAL_NOISE = 0.10  # bare "temporal_phases"
DEFAULT_TEMPORAL_PHASES = 8
DEFAULT_STALE_HITRATE = 0.5  # bare "stale_resident" (target cache hit rate)
DEFAULT_STRIDED_STRIDE = 9973  # bare "strided_sweep"
DEFAULT_SAWTOOTH_PERIOD = 0  # bare "sawtooth" => period == num_bins
DEFAULT_SAWTOOTH_STRIDE = 1
DEFAULT_SAWTOOTH_SCATTER = False
# Representative single-channel EVEN cache slot count for CHARACTERIZATION FIGURES only
# (the real generator queries the compiled policy's runtime S per binary). Matches B200
# single-channel EVEN and is passed explicitly through the same generator interface.
CHAR_CACHE_SLOTS = 8192

# Representative single-channel poison schedule and the cache hashes it targets. These
# mirror the C++ generator. The characterization helper has no binary/channel context,
# so it deliberately uses the same single-channel EVEN slot count as stale_resident.
POISON_SINGLE_WINDOW = 1 << 24
POISON_SINGLE_CLAIM_PREFIX = 5 << 16
POISON_MULTI_WINDOW = 1 << 26
POISON_MULTI_CLAIM_PREFIX = 5 << 18


def _kernel_hash_configuration() -> tuple[int, int, int]:
    """Read the cache hash defaults from the C++ source of truth."""
    source = (
        Path(__file__).resolve().parents[3] / "cub/detail/histogram_cache_hash.cuh"
    ).read_text(encoding="utf-8")

    def uint_constant(name: str) -> int:
        match = re.search(rf"\b{name}\s*=\s*(\d+)u\s*;", source)
        if match is None:
            raise RuntimeError(f"cannot find {name} in histogram_cache_hash.cuh")
        return int(match.group(1))

    mode_match = re.search(r"#\s*define\s+CUB_HISTO_CACHE_HASH_MODE\s+(\d+)", source)
    if mode_match is None:
        raise RuntimeError(
            "cannot find CUB_HISTO_CACHE_HASH_MODE in histogram_cache_hash.cuh"
        )
    mode = int(os.environ.get("CUB_HISTO_CACHE_HASH_MODE", mode_match.group(1)))
    return (
        mode,
        uint_constant("cache_primary_hash_multiplier"),
        uint_constant("cache_secondary_hash_multiplier"),
    )


CACHE_HASH_MODE, POISON_PRIMARY_MULTIPLIER, POISON_SECONDARY_MULTIPLIER = (
    _kernel_hash_configuration()
)
TEMPORAL_NOISE_SELECT_SALT = 0x74656D706F72616C
TEMPORAL_NOISE_VALUE_SALT = 0x6A69747465725661

_U64 = np.uint64
_MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# Bit-exact RNG / index helpers (splitmix64 + LCG decorrelation), mirroring the
# __host__ __device__ inline functions in the header. All arithmetic is done in
# explicit uint64 with wraparound so results match the C++ byte-for-byte.
# ---------------------------------------------------------------------------


def u01_from_hash(x: np.ndarray | int) -> np.ndarray:
    """Port of u01_from_hash: splitmix64 finalizer -> double in [0, 1)."""
    x = np.asarray(x, dtype=_U64)
    with np.errstate(over="ignore"):
        x = x + _U64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> _U64(27))) * _U64(0x94D049BB133111EB)
        x = x ^ (x >> _U64(31))
    return (x >> _U64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)  # 2^53


def element_key(i: np.ndarray | int, seed: int) -> np.ndarray:
    """Port of element_key: i*6364136223846793005 + 1442695040888963407 + seed*phi."""
    i = np.asarray(i, dtype=_U64)
    with np.errstate(over="ignore"):
        return (
            i * _U64(6364136223846793005)
            + _U64(1442695040888963407)
            + _U64(seed & _MASK64) * _U64(0x9E3779B97F4A7C15)
        )


def scatter_bin(rank: int, num_bins: int, offset: int) -> int:
    """Port of scatter_bin: affine coprime permutation, evaluated in uint64."""
    return int((((rank & _MASK64) * K_SCATTER_PRIME + offset) & _MASK64) % num_bins)


def _feistel_mix(x: np.ndarray, k: int) -> np.ndarray:
    """Port of feistel_mix: splitmix64-style round function (uint64 wraparound)."""
    with np.errstate(over="ignore"):
        z = x + _U64(k & _MASK64) + _U64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB)
        return z ^ (z >> _U64(31))


def permute_index(i: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Port of permute_index: a keyed pseudo-random bijection on [0, n) via a
    4-round balanced Feistel network with cycle-walking. Vectorized over `i`;
    bit-for-bit identical to the device functor (same half-width, mask, rounds,
    round keys, and mixing constants)."""
    i = np.asarray(i, dtype=_U64)
    if n <= 1:
        return np.zeros_like(i)
    bits = int(n - 1).bit_length()  # ceil(log2(n)) for n >= 2
    half = (bits + 1) // 2
    mask = _MASK64 if half >= 64 else ((1 << half) - 1)
    mask = _U64(mask)
    hh = _U64(half)
    x = i.copy()
    todo = np.ones(x.shape, dtype=bool)
    with np.errstate(over="ignore"):
        for _ in range(128):  # cycle-walk; expected < 4 iters
            xs = x[todo]
            left = (xs >> hh) & mask
            right = xs & mask
            for rnd in range(4):
                new_left = right
                new_right = left ^ (_feistel_mix(right, seed + rnd) & mask)
                left, right = new_left, new_right
            x[todo] = (left << hh) | right
            todo = x >= _U64(n)
            if not todo.any():
                break
    return x


def _scatter_bin_vec(ranks: np.ndarray, num_bins: int, offset: int) -> np.ndarray:
    r = ranks.astype(object)  # python big-ints to avoid overflow before mod
    return np.array(
        [((int(x) * K_SCATTER_PRIME + offset) & _MASK64) % num_bins for x in r],
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# Spec + parse — mirror ShapeSpec / parse_input_shape / knob_or.
# ---------------------------------------------------------------------------

VALID_SHAPES = {
    "concentrated",
    "powerlaw",
    "hash_synonym",
    "stale_resident",
    "temporal_phases",
    "strided_sweep",
    "sawtooth",
    "poison",
}


@dataclass
class ShapeSpec:
    shape: str
    knob: float = 0.0
    has_knob: bool = False
    knob2: float = 0.0
    has_knob2: bool = False
    knob3: float = 0.0
    has_knob3: bool = False


def parse_input_shape(spec: str) -> ShapeSpec:
    """Port of parse_input_shape: ``name[:knob[:knob2[:knob3]]]``."""
    parts = spec.split(":")
    if len(parts) > 4:
        raise ValueError(f"Too many InputShape parameters: {spec}")
    name = parts[0]
    knobs = [float(value) for value in parts[1:]]
    if name not in VALID_SHAPES:
        raise ValueError(f"Unknown InputShape: {spec}")
    return ShapeSpec(
        shape=name,
        knob=knobs[0] if len(knobs) > 0 else 0.0,
        has_knob=len(knobs) > 0,
        knob2=knobs[1] if len(knobs) > 1 else 0.0,
        has_knob2=len(knobs) > 1,
        knob3=knobs[2] if len(knobs) > 2 else 0.0,
        has_knob3=len(knobs) > 2,
    )


def knob_or(spec: ShapeSpec, default_value: float) -> float:
    return spec.knob if spec.has_knob else default_value


# ---------------------------------------------------------------------------
# pmf math — mirror normalized_entropy / ranked_powerlaw / solvers.
# ---------------------------------------------------------------------------


def normalized_entropy(pmf: np.ndarray) -> float:
    if len(pmf) <= 1:
        return 0.0
    p = pmf[pmf > 0]
    return float(-(p * np.log2(p)).sum() / np.log2(len(pmf)))


def ranked_powerlaw(num_bins: int, s: float) -> np.ndarray:
    w = np.arange(1, num_bins + 1, dtype=np.float64) ** (-s)
    return w / w.sum()


def solve_powerlaw_exponent(num_bins: int, target: float) -> float:
    """Bisection mirror: 60 iters, [0, 60], entropy decreasing in s."""
    lo, hi = 0.0, 60.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if normalized_entropy(ranked_powerlaw(num_bins, mid)) < target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def softmax_logits(num_bins: int, seed: int) -> np.ndarray:
    """Per-bin standard-normal logits (Box-Muller on two hashed uniforms),
    mirroring softmax_logits() in the header."""
    b = np.arange(num_bins, dtype=_U64)
    u1 = u01_from_hash(element_key(b, seed))
    u2 = u01_from_hash(element_key(b, seed ^ 0xD1B54A32D192ED03))
    u1 = np.where(u1 > 0.0, u1, 1e-300)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def softmax_pmf(g: np.ndarray, T: float) -> np.ndarray:
    z = (g - g.max()) / T
    e = np.exp(z)
    return e / e.sum()


def solve_softmax_pmf(num_bins: int, target: float, seed: int) -> np.ndarray:
    """Softmax over random logits, temperature bisected in log-space to hit the
    target normalized entropy (entropy increases with T). Mirrors the header."""
    g = softmax_logits(num_bins, seed)
    lo, hi = 1e-3, 1e3
    for _ in range(80):
        mid = (lo * hi) ** 0.5
        if normalized_entropy(softmax_pmf(g, mid)) < target:
            lo = mid
        else:
            hi = mid
    return softmax_pmf(g, (lo * hi) ** 0.5)


# ---------------------------------------------------------------------------
# build_pmf — mirror the C++ switch exactly (i.i.d. distribution shapes).
# ---------------------------------------------------------------------------


def build_pmf(
    spec: ShapeSpec, num_bins: int, seed: int, cache_slots: int
) -> np.ndarray:
    pmf = np.zeros(num_bins, dtype=np.float64)
    offset = seed % num_bins

    if spec.shape == "concentrated":
        target = knob_or(spec, DEFAULT_CONCENTRATED_ENTROPY)
        if target >= 1.0:
            pmf[:] = 1.0 / num_bins
        elif target <= 0.0:
            pmf[scatter_bin(0, num_bins, offset)] = 1.0
        else:
            # Completely random bin probabilities dialed to the target entropy
            # (softmax over per-bin random logits). All bins occupied, smoothly
            # more uniform as the knob -> 1; no single dominant "hot" bin.
            pmf[:] = solve_softmax_pmf(num_bins, target, seed)

    elif spec.shape == "powerlaw":
        s = solve_powerlaw_exponent(num_bins, knob_or(spec, DEFAULT_POWERLAW_ENTROPY))
        w = ranked_powerlaw(num_bins, s)
        idx = _scatter_bin_vec(np.arange(num_bins), num_bins, offset)
        pmf[idx] = w

    elif spec.shape == "hash_synonym":
        hot_share = knob_or(spec, DEFAULT_HASH_SYNONYM_HOTSHARE)
        syn = find_hash_synonyms(num_bins, cache_slots, seed)
        pmf[:] = (1.0 - hot_share) / num_bins
        if syn:
            per = hot_share / len(syn)
            for b in syn:
                pmf[b] += per
        else:
            pmf[scatter_bin(0, num_bins, offset)] += hot_share

    else:
        raise ValueError(f"build_pmf called with non-distribution shape {spec.shape!r}")

    return pmf


def is_ordering_shape(shape: str) -> bool:
    return shape in (
        "hash_synonym",
        "stale_resident",
        "temporal_phases",
        "strided_sweep",
        "sawtooth",
        "poison",
    )


# Near-resident branch table (mirrors kStaleNearRate / kStaleNearR in the header).
_STALE_NEAR_RATE = [1.000, 0.990, 0.944, 0.888, 0.800, 0.744]
_STALE_NEAR_R = [0.300, 0.519, 0.697, 0.898, 1.126, 1.344]
_STALE_SPREAD_X = 0.744


def stale_working_set(target_hit_rate: float, cache_slots: int, num_bins: int) -> int:
    """Port of stale_working_set: W for a target cache hit rate X against S slots.
    W == round(S / X) for X <= 0.744 (perfect-spread regime, measured), else invert the
    near-resident table. See the header for the calibration."""
    S = float(cache_slots if cache_slots > 0 else 1)
    X = min(max(target_hit_rate, 1e-3), 0.999)
    if X <= _STALE_SPREAD_X:
        r = 1.0 / X
    elif X >= _STALE_NEAR_RATE[0]:
        r = _STALE_NEAR_R[0]
    else:
        r = _STALE_NEAR_R[-1]
        for i in range(len(_STALE_NEAR_RATE) - 1):
            if X <= _STALE_NEAR_RATE[i] and X > _STALE_NEAR_RATE[i + 1]:
                t = (_STALE_NEAR_RATE[i] - X) / (
                    _STALE_NEAR_RATE[i] - _STALE_NEAR_RATE[i + 1]
                )
                r = _STALE_NEAR_R[i] + t * (_STALE_NEAR_R[i + 1] - _STALE_NEAR_R[i])
                break
    return max(1, min(int(round(r * S)), num_bins))


def cache_slot(bin_index: int, slots: int, multiplier: int) -> int:
    """Mirror CUB's source-derived cache_slot_from_hash configuration."""
    if slots <= 0 or slots & (slots - 1):
        raise ValueError("histogram input cache slots must be a positive power of two")
    slot_bits = slots.bit_length() - 1
    product = (bin_index * multiplier) & 0xFFFFFFFF
    if CACHE_HASH_MODE == 1:
        return product >> (32 - slot_bits)
    if CACHE_HASH_MODE == 2:
        return ((product >> 15) ^ product) & (slots - 1)
    if CACHE_HASH_MODE == 0:
        return product & (slots - 1)
    raise ValueError(f"unsupported CUB_HISTO_CACHE_HASH_MODE={CACHE_HASH_MODE}")


def resolve_cache_slots(cache_slots: int) -> int:
    """Honor the same generator override names as the C++ implementation."""
    forced = os.environ.get("CUB_HISTO_INPUT_CACHE_SLOTS") or os.environ.get(
        "CUB_HISTO_STALE_SLOTS"
    )
    if forced is not None and int(forced) > 0:
        return int(forced)
    return cache_slots


def find_hash_synonyms(num_bins: int, slots: int, seed: int) -> list[int]:
    """Find up to 32 bins sharing one real primary cache slot."""
    occupancy = np.zeros(slots, dtype=np.int64)
    for bin_index in range(num_bins):
        occupancy[cache_slot(bin_index, slots, POISON_PRIMARY_MULTIPLIER)] += 1
    max_occupancy = int(occupancy.max())
    start = seed % slots
    target = next(
        (start + delta) & (slots - 1)
        for delta in range(slots)
        if occupancy[(start + delta) & (slots - 1)] == max_occupancy
    )
    result = []
    for bin_index in range(num_bins):
        if cache_slot(bin_index, slots, POISON_PRIMARY_MULTIPLIER) == target:
            result.append(bin_index)
            if len(result) == K_HASH_SYNONYM_COUNT:
                break
    return result


def make_hash_synonym_bin_set(
    num_bins: int, slots: int, seed: int
) -> tuple[list[int], list[int], bool]:
    """Return hot synonyms and cache-priming claims, mirroring the C++ helper."""
    synonyms = find_hash_synonyms(num_bins, slots, seed)
    if not synonyms:
        return [], [], False

    primary_slot = cache_slot(synonyms[0], slots, POISON_PRIMARY_MULTIPLIER)
    claims = [synonyms[0]]
    blocked_secondary = set()
    synonym_set = set(synonyms)
    primary_blocker = [-1] * slots
    for bin_index in range(num_bins):
        if bin_index not in synonym_set:
            slot = cache_slot(bin_index, slots, POISON_PRIMARY_MULTIPLIER)
            if primary_blocker[slot] < 0:
                primary_blocker[slot] = bin_index
    for synonym in synonyms[1:]:
        secondary_slot = cache_slot(synonym, slots, POISON_SECONDARY_MULTIPLIER)
        if secondary_slot == primary_slot or secondary_slot in blocked_secondary:
            continue
        blocker = primary_blocker[secondary_slot]
        if blocker < 0 or len(claims) >= K_HASH_SYNONYM_COUNT:
            return [], [], False
        claims.append(blocker)
        blocked_secondary.add(secondary_slot)
    return synonyms, claims, True


def find_poison_bins(num_bins: int, slots: int) -> tuple[int, int, int, bool]:
    """Find blocker0, blocker1, poison, valid exactly as the C++ host helper does."""
    fallback = (0, 0, max(0, num_bins - 1), False)
    if num_bins <= slots or slots <= 0 or slots & (slots - 1):
        return fallback

    owner0 = [-1] * slots
    owner1 = [-1] * slots
    for bin_index in range(num_bins):
        slot = cache_slot(bin_index, slots, POISON_PRIMARY_MULTIPLIER)
        if owner0[slot] == -1:
            owner0[slot] = bin_index
        elif owner1[slot] == -1:
            owner1[slot] = bin_index

    for poison in range(num_bins - 1, -1, -1):
        primary_slot = cache_slot(poison, slots, POISON_PRIMARY_MULTIPLIER)
        secondary_slot = cache_slot(poison, slots, POISON_SECONDARY_MULTIPLIER)
        blocker0 = (
            owner1[primary_slot]
            if owner0[primary_slot] == poison
            else owner0[primary_slot]
        )
        if blocker0 < 0:
            continue
        blocker1 = (
            blocker0 if secondary_slot == primary_slot else owner0[secondary_slot]
        )
        if blocker1 < 0:
            continue
        return blocker0, blocker1, poison, True
    return fallback


# ---------------------------------------------------------------------------
# Core generation — mirror generate_shape_impl. Returns BIN INDICES (the value
# mapping is applied by the public entry points below).
# ---------------------------------------------------------------------------


def _bins_for_spec(
    spec: ShapeSpec,
    n: int,
    num_bins: int,
    seed: int,
    cache_slots: int,
    sample_stride: int,
) -> np.ndarray:
    if sample_stride <= 0:
        raise ValueError("histogram input sample stride must be positive")
    cache_slots = resolve_cache_slots(cache_slots)
    offset = seed % num_bins

    def channel_indices(channel: int) -> np.ndarray:
        return np.arange(channel, n, sample_stride, dtype=np.int64)

    def channel_seed(channel: int) -> int:
        return (seed + channel * K_CHANNEL_SEED_STRIDE) & _MASK64

    # Exact-uniform endpoint: exact round-robin counts, but in pseudo-random
    # sequence order (a Feistel permutation of the tiling). Uniform counts,
    # randomly distributed in the input -- NOT the sequential ramp (= sawtooth).
    if spec.shape == "concentrated" and spec.has_knob and spec.knob >= 1.0:
        bins = np.empty(n, dtype=np.int64)
        for channel in range(sample_stride):
            indices = channel_indices(channel)
            perm = permute_index(
                np.arange(indices.size, dtype=_U64),
                indices.size,
                channel_seed(channel),
            )
            bins[indices] = (perm % np.uint64(num_bins)).astype(np.int64)
        return bins

    if not is_ordering_shape(spec.shape):
        pmf = build_pmf(spec, num_bins, seed, cache_slots)
        cdf = np.cumsum(pmf)
        cdf[-1] = 1.0  # guard fp drift, matches header
        bins = np.empty(n, dtype=np.int64)
        for channel in range(sample_stride):
            indices = channel_indices(channel)
            u = u01_from_hash(
                element_key(np.arange(indices.size, dtype=_U64), channel_seed(channel))
            )
            bins[indices] = np.searchsorted(cdf, u, side="right").astype(np.int64)
        np.clip(bins, 0, num_bins - 1, out=bins)
        return bins

    if spec.shape == "hash_synonym":
        synonyms, claims, enabled = make_hash_synonym_bin_set(
            num_bins, cache_slots, seed
        )
        hot_share = min(1.0, max(0.0, knob_or(spec, DEFAULT_HASH_SYNONYM_HOTSHARE)))
        bins = np.empty(n, dtype=np.int64)
        index = np.arange(n, dtype=_U64)
        for channel in range(sample_stride):
            indices = channel_indices(channel)
            samples = np.arange(indices.size, dtype=_U64)
            stream_seed = channel_seed(channel)
            select = u01_from_hash(
                element_key(samples, stream_seed ^ 0x6861736853796E31)
            )
            value = u01_from_hash(
                element_key(samples, stream_seed ^ 0x6861736853796E32)
            )
            channel_bins = (value * float(num_bins)).astype(np.int64)
            np.clip(channel_bins, 0, num_bins - 1, out=channel_bins)
            if enabled:
                synonym_array = np.asarray(synonyms, dtype=np.int64)
                hot = select < hot_share
                hot_index = (value[hot] * float(len(synonyms))).astype(np.int64)
                np.clip(hot_index, 0, len(synonyms) - 1, out=hot_index)
                channel_bins[hot] = synonym_array[hot_index]
            bins[indices] = channel_bins

        if enabled:
            multi_layout = sample_stride > 1
            window = POISON_MULTI_WINDOW if multi_layout else POISON_SINGLE_WINDOW
            claim_prefix = (
                POISON_MULTI_CLAIM_PREFIX
                if multi_layout
                else POISON_SINGLE_CLAIM_PREFIX
            )
            claim = (index & _U64(window - 1)) < _U64(claim_prefix)
            claim_array = np.asarray(claims, dtype=np.int64)
            sample = index // _U64(sample_stride)
            bins[claim] = claim_array[
                (sample[claim] % _U64(len(claims))).astype(np.int64)
            ]
        return bins

    if spec.shape == "strided_sweep":
        stride = (
            int(round(spec.knob)) & _MASK64 if spec.has_knob else DEFAULT_STRIDED_STRIDE
        )
        samples = np.arange(n, dtype=np.int64) // sample_stride
        return np.array(
            [((int(x) * stride) & _MASK64) % num_bins for x in samples],
            dtype=np.int64,
        )

    if spec.shape == "sawtooth":
        # rank = (stride*i) % period, optionally scattered. Masking the parsed
        # integers to uint64 mirrors the C++ casts and NumPy uint64 multiply wraps
        # exactly like the device expression before the modulo.
        requested = (
            (int(round(spec.knob)) & _MASK64)
            if spec.has_knob
            else DEFAULT_SAWTOOTH_PERIOD
        )
        period = num_bins if requested == 0 else min(requested, num_bins)
        stride = (
            (int(round(spec.knob2)) & _MASK64)
            if spec.has_knob2
            else DEFAULT_SAWTOOTH_STRIDE
        )
        scatter = (
            (round(spec.knob3) != 0) if spec.has_knob3 else DEFAULT_SAWTOOTH_SCATTER
        )
        samples = np.arange(n, dtype=_U64) // _U64(sample_stride)
        rank = (samples * _U64(stride)) % _U64(period)
        if scatter:
            return num_bins - 1 - rank.astype(np.int64)
        return rank.astype(np.int64)

    if spec.shape == "poison":
        blocker0, blocker1, poison, enabled = find_poison_bins(num_bins, cache_slots)
        bins = np.full(n, poison, dtype=np.int64)
        if enabled:
            index = np.arange(n, dtype=_U64)
            multi_layout = sample_stride > 1
            window = POISON_MULTI_WINDOW if multi_layout else POISON_SINGLE_WINDOW
            claim_prefix = (
                POISON_MULTI_CLAIM_PREFIX
                if multi_layout
                else POISON_SINGLE_CLAIM_PREFIX
            )
            claim = (index & _U64(window - 1)) < _U64(claim_prefix)
            claim_index = index[claim]
            bins[claim] = np.where(
                ((claim_index // _U64(sample_stride)) & _U64(1)) != 0,
                blocker1,
                blocker0,
            )
        return bins

    if spec.shape == "temporal_phases":
        noise = min(1.0, max(0.0, knob_or(spec, DEFAULT_TEMPORAL_NOISE)))
        requested = (
            int(round(spec.knob2)) if spec.has_knob2 else DEFAULT_TEMPORAL_PHASES
        )
        phases = max(1, min(requested, num_bins))
        bins = np.empty(n, dtype=np.int64)
        for channel in range(sample_stride):
            indices = channel_indices(channel)
            samples = np.arange(indices.size, dtype=np.int64)
            phase = (samples * phases) // indices.size
            np.clip(phase, 0, phases - 1, out=phase)
            rank = (phase.astype(object) * num_bins) // phases
            channel_bins = _scatter_bin_vec(rank, num_bins, offset)
            if noise > 0.0:
                stream_seed = channel_seed(channel)
                select = u01_from_hash(
                    element_key(
                        samples.astype(_U64),
                        stream_seed ^ TEMPORAL_NOISE_SELECT_SALT,
                    )
                )
                noisy = select < noise
                if noisy.any():
                    u = u01_from_hash(
                        element_key(
                            samples[noisy].astype(_U64),
                            stream_seed ^ TEMPORAL_NOISE_VALUE_SALT,
                        )
                    )
                    random_bins = (u * float(num_bins)).astype(np.int64)
                    np.clip(random_bins, 0, num_bins - 1, out=random_bins)
                    channel_bins[noisy] = random_bins
            bins[indices] = channel_bins
        return bins

    if spec.shape == "stale_resident":
        # A cold working set of W distinct bins vs a per-block no-eviction cache of S
        # slots, sized for a target hit rate X (the knob). W is solved from S (see
        # stale_working_set); the per-element key is a UNIFORM hash of the index (NOT a
        # cyclic i%span counter) so every block sees the whole working set. Mirrors the
        # header's stale_functor + stale_working_set. For figures S is the representative
        # single-channel EVEN slot count (the real generator queries the runtime S).
        target = knob_or(spec, DEFAULT_STALE_HITRATE)
        span = stale_working_set(target, cache_slots, num_bins)
        bins = np.empty(n, dtype=np.int64)
        for channel in range(sample_stride):
            indices = channel_indices(channel)
            u = u01_from_hash(
                element_key(
                    np.arange(indices.size, dtype=_U64),
                    channel_seed(channel) ^ 0x5ADE57A1E00FF5E7,
                )
            )
            k = (u * float(span)).astype(np.int64)
            np.clip(k, 0, span - 1, out=k)
            bins[indices] = num_bins - 1 - k
        return bins

    raise ValueError(f"unreachable shape {spec.shape!r}")


# ---------------------------------------------------------------------------
# Bin -> sample value mappers — mirror even_bin_to_value / range_bin_to_value.
# ---------------------------------------------------------------------------


def _even_bin_to_value(bins, num_bins, lo, hi, dtype):
    w = (float(hi) - float(lo)) / num_bins
    v = float(lo) + (bins.astype(np.float64) + 0.5) * w
    if np.issubdtype(np.dtype(dtype), np.integer):
        v = np.floor(v)
        v = np.clip(v, lo, hi - 1)
        return v.astype(dtype)
    v = np.maximum(v, float(lo))
    return v.astype(dtype)


def _range_bin_to_value(bins, levels, dtype):
    num_bins = len(levels) - 1
    b = np.clip(bins, 0, num_bins - 1)
    loi = levels[b].astype(np.float64)
    hii = levels[b + 1].astype(np.float64)
    v = 0.5 * (loi + hii)
    if np.issubdtype(np.dtype(dtype), np.integer):
        v = np.floor(v)
    s = v.astype(dtype)
    # guarantee s in [levels[b], levels[b+1]) like the header
    below = s < levels[b]
    s[below] = levels[b][below]
    atorabove = s >= levels[b + 1]
    s[atorabove] = levels[b][atorabove]
    return s


# ---------------------------------------------------------------------------
# Public entry points — mirror generate_histogram_input_{even,range}.
# Return (values, bins). `bins` is what the histogram counts; graphs use it.
# ---------------------------------------------------------------------------


def generate_histogram_input_even(
    spec,
    n,
    num_bins,
    lower,
    upper,
    dtype=np.int32,
    seed=42,
    cache_slots=CHAR_CACHE_SLOTS,
    sample_stride=1,
):
    if isinstance(spec, str):
        spec = parse_input_shape(spec)
    bins = _bins_for_spec(spec, n, num_bins, seed, cache_slots, sample_stride)
    values = _even_bin_to_value(bins, num_bins, lower, upper, dtype)
    return values, bins


def generate_histogram_input_range(
    spec,
    n,
    num_bins,
    levels,
    dtype=np.int32,
    seed=42,
    cache_slots=CHAR_CACHE_SLOTS,
    sample_stride=1,
):
    if isinstance(spec, str):
        spec = parse_input_shape(spec)
    levels = np.asarray(levels)
    bins = _bins_for_spec(spec, n, num_bins, seed, cache_slots, sample_stride)
    values = _range_bin_to_value(bins, levels, dtype)
    return values, bins


# Convenience for the viz scripts: bins only (EVEN-equivalent), no value map.
def generate_bins(
    spec,
    n,
    num_bins,
    seed=42,
    cache_slots=CHAR_CACHE_SLOTS,
    sample_stride=1,
):
    if isinstance(spec, str):
        spec = parse_input_shape(spec)
    return _bins_for_spec(spec, n, num_bins, seed, cache_slots, sample_stride)


# ---------------------------------------------------------------------------
# Self-test: confirm the mirror reproduces the documented behavior.
# ---------------------------------------------------------------------------


def _assert_contracts():
    """Golden contracts shared by the figures and benchmark generator tests."""
    if K_SCATTER_PRIME != 2147483647:
        raise AssertionError("power-law scatter placement changed")

    for num_bins, expected in (
        (256, [42, 41, 40, 39, 38, 37, 36, 35]),
        (100, [42, 89, 36, 83, 30, 77, 24, 71]),
    ):
        pmf = build_pmf(parse_input_shape("powerlaw:0.5"), num_bins, 42, 8192)
        observed = np.argsort(-pmf, kind="stable")[: len(expected)].tolist()
        if observed != expected:
            raise AssertionError(
                f"powerlaw bin placement changed at B={num_bins}: {observed}"
            )

    for entropy, exponent, top_share in (
        (0.75, 1.0495426693985075, 0.1841284718868535),
        (0.50, 1.4799203229256670, 0.3922707569591881),
        (0.25, 2.1519418618806565, 0.6574873803199386),
    ):
        actual_exponent = solve_powerlaw_exponent(256, entropy)
        weights = ranked_powerlaw(256, actual_exponent)
        if abs(actual_exponent - exponent) >= 1e-12:
            raise AssertionError(f"powerlaw exponent changed for entropy={entropy}")
        if abs(float(weights[0]) - top_share) >= 1e-12:
            raise AssertionError(f"powerlaw top share changed for entropy={entropy}")

    channels, samples, num_bins = 4, 65536, 256
    multi = generate_bins(
        "powerlaw:0.25",
        channels * samples,
        num_bins,
        seed=42,
        sample_stride=channels,
    )
    expected_top = [42, 41, 40, 39, 38, 37, 36, 35]
    for channel in range(channels):
        counts = np.bincount(multi[channel::channels], minlength=num_bins)
        observed = np.argsort(-counts, kind="stable")[:8].tolist()
        if observed != expected_top:
            raise AssertionError(
                f"multi channel {channel} remapped powerlaw bins: {observed}"
            )
        if abs(normalized_entropy(counts / counts.sum()) - 0.25) >= 0.01:
            raise AssertionError(f"multi channel {channel} changed powerlaw entropy")

    for shape in ("concentrated:1.0", "strided_sweep", "sawtooth"):
        bins = generate_bins(
            shape,
            channels * samples,
            num_bins,
            seed=42,
            sample_stride=channels,
        )
        for channel in range(channels):
            counts = np.bincount(bins[channel::channels], minlength=num_bins)
            if not np.all(counts == samples // num_bins):
                raise AssertionError(
                    f"multi channel {channel} does not cover {shape} exactly"
                )

    for bins_count in (65536, 60000):
        stale = generate_bins(
            "stale_resident:0.5",
            200000,
            bins_count,
            seed=42,
            cache_slots=8192,
        )
        if int(stale.min()) < bins_count - 16384 or int(stale.max()) != bins_count - 1:
            raise AssertionError(
                f"stale_resident escaped its upper window at B={bins_count}"
            )

    saw = generate_bins("sawtooth:8192:2654435761:1", 8192, 60000, seed=42)
    if int(saw.min()) != 60000 - 8192 or int(saw.max()) != 59999:
        raise AssertionError("scattered sawtooth escaped its upper window")
    if np.unique(saw).size != 8192:
        raise AssertionError("scattered sawtooth support cardinality changed")

    multi_saw = generate_bins(
        "sawtooth:8192:2654435761:1",
        channels * 8192,
        16384,
        seed=42,
        sample_stride=channels,
    )
    for channel in range(channels):
        channel_saw = multi_saw[channel::channels]
        if np.unique(channel_saw).size != 8192 or int(channel_saw.min()) != 8192:
            raise AssertionError(
                f"multi channel {channel} lost scattered sawtooth support"
            )

    phases = generate_bins("temporal_phases:0.10:8", 200000, 256, seed=42)
    per_phase = len(phases) // 8
    observed_phases = [
        int(
            np.bincount(
                phases[phase * per_phase : (phase + 1) * per_phase],
                minlength=256,
            ).argmax()
        )
        for phase in range(8)
    ]
    if observed_phases != [42, 10, 234, 202, 170, 138, 106, 74]:
        raise AssertionError(f"temporal phase centers changed: {observed_phases}")

    synonyms, claims, enabled = make_hash_synonym_bin_set(32768, 1024, 42)
    primary_slots = {
        cache_slot(bin_index, 1024, POISON_PRIMARY_MULTIPLIER) for bin_index in synonyms
    }
    claimed_slots = {
        cache_slot(bin_index, 1024, POISON_PRIMARY_MULTIPLIER) for bin_index in claims
    }
    secondary_slots = {
        cache_slot(bin_index, 1024, POISON_SECONDARY_MULTIPLIER)
        for bin_index in synonyms[1:]
    }
    if (
        not enabled
        or len(synonyms) != K_HASH_SYNONYM_COUNT
        or len(primary_slots) != 1
        or not secondary_slots.issubset(claimed_slots)
    ):
        raise AssertionError("hash_synonym no longer blocks both cache probes")

    single_poison = generate_bins(
        "poison", 400000, 32768, cache_slots=1024, sample_stride=1
    )
    poison = find_poison_bins(32768, 1024)[2]
    if (
        int(np.count_nonzero(single_poison == poison))
        != 400000 - POISON_SINGLE_CLAIM_PREFIX
    ):
        raise AssertionError("single-channel poison schedule changed")

    multi_poison = generate_bins(
        "poison",
        channels * 400000,
        32768,
        cache_slots=8192,
        sample_stride=channels,
    )
    poison = find_poison_bins(32768, 8192)[2]
    if int(np.count_nonzero(multi_poison == poison)) != (
        channels * 400000 - POISON_MULTI_CLAIM_PREFIX
    ):
        raise AssertionError("multi-channel poison schedule changed")


def _demo():
    N, B = 200_000, 64

    print("== exact endpoints ==")
    b = generate_bins("concentrated:1.0", N, B, seed=42)
    c = np.bincount(b, minlength=B)
    print(
        f"  concentrated:1.0  per-bin min={c.min()} max={c.max()} (exact uniform), H={normalized_entropy(c / N):.4f}"
    )
    b = generate_bins("concentrated:0.0", N, B, seed=42)
    c = np.bincount(b, minlength=B)
    print(
        f"  concentrated:0.0  nonzero bins={np.count_nonzero(c)} at bin {c.argmax()} (constant, scattered off 0)"
    )

    print("== concentrated entropy knob (measured top-bin share) ==")
    for e in (1.0, 0.75, 0.5, 0.25, 0.0):
        b = generate_bins(f"concentrated:{e}", N, B, seed=42)
        c = np.bincount(b, minlength=B)
        print(
            f"  E={e:.2f}: H={normalized_entropy(c / N):.3f}  top-share={c.max() / N:.1%}"
        )

    print("== hot bin varies with seed (concentrated:0.3) ==")
    args = {
        int(
            np.bincount(
                generate_bins("concentrated:0.3", N, B, seed=s), minlength=B
            ).argmax()
        )
        for s in range(1, 6)
    }
    print(f"  argmax bins across seeds 1..5: {sorted(args)} (not pinned to 0)")

    print("== adversarial: hash_synonym collides on one slot after its claim prefix ==")
    b = generate_bins("hash_synonym", max(N, 4_000_000), 262144, seed=42)
    hotbins = np.argsort(np.bincount(b, minlength=262144))[::-1][:K_HASH_SYNONYM_COUNT]
    print(
        f"  distinct primary slots among the {K_HASH_SYNONYM_COUNT} hottest bins: "
        f"{len(set(cache_slot(int(x), CHAR_CACHE_SLOTS, POISON_PRIMARY_MULTIPLIER) for x in hotbins))} (want 1)"
    )

    print("== adversarial: temporal_phases moves the hot bin with 10% noise ==")
    b = generate_bins("temporal_phases:0.10", N, 256, seed=42)
    q = N // DEFAULT_TEMPORAL_PHASES
    print(
        "  per-phase hot bin: "
        f"{[int(np.bincount(b[i * q : (i + 1) * q], minlength=256).argmax()) for i in range(DEFAULT_TEMPORAL_PHASES)]}"
    )


if __name__ == "__main__":
    _assert_contracts()
    print("== distribution contracts: passed ==")
    _demo()
