#!/usr/bin/env python3
"""Bit-exact Python mirror of the CUB histogram input-shape generators.

SOURCE OF TRUTH: cub/benchmarks/bench/histogram/histogram_inputs.cuh (on `main`).
This module is a faithful host-side port of that header, NOT an independent
design sketch. Every shape, knob, default, and the RNG/scatter math below mirror
the C++ exactly, so the bin indices produced here equal what the benchmark
produces on device for the same (shape, n, num_bins, seed). Keep them in sync:
if the .cuh changes, update this file.

The C++ replaced the legacy bitwise-AND "entropy" knob with a tunable
`InputShape` axis whose values are `name[:knob]`. There is ONE `concentrated`
shape spanning uniform<->constant via its entropy knob (no separate
uniform/constant/spike names), plus multi-hot (powerlaw/zipf) and
cache-adversarial (hash_synonym/stale_resident) and ordering
(temporal_phases/strided_sweep/sawtooth) shapes.

Mechanism mirrored from the header:
  * Every shape decides a per-element BIN index in [0, num_bins); a bin->value
    mapper then emits a SampleT in the bin's interval (EVEN: midpoint; RANGE:
    level-interval midpoint). CUB re-derives the bin, so the in-bench verifier
    validates the mapping.
  * i.i.d. distribution shapes: build a host pmf -> inclusive CDF -> per element
    draw u01_from_hash(element_key(i, seed)) and upper_bound_cdf into a bin.
  * ordering shapes (stale_resident/temporal_phases/strided_sweep): positional
    functors, order is intrinsic.
  * the concentrated uniform endpoint (knob>=1.0) is an exact equal-count tiling
    bin(i)=i%num_bins (sequential, NOT shuffled).
  * hot bins are scattered off zero via scatter_bin (a fixed coprime
    permutation seeded by seed%num_bins), so the mode is never bin 0.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# Constants — mirror histogram_inputs.cuh exactly.
# ---------------------------------------------------------------------------
K_ADVERSARIAL_CACHE_SLOTS = 4096           # kAdversarialCacheSlots
K_HASH_SYNONYM_COUNT      = 32             # kHashSynonymCount
K_SCATTER_PRIME           = 2147483647     # kScatterPrime (2^31 - 1)

DEFAULT_CONCENTRATED_ENTROPY = 0.5         # bare "concentrated"
DEFAULT_POWERLAW_ENTROPY     = 0.5         # bare "powerlaw"
DEFAULT_ZIPF_EXPONENT        = 1.0         # bare "zipf"
DEFAULT_HASH_SYNONYM_HOTSHARE = 0.9        # bare "hash_synonym"
DEFAULT_TEMPORAL_PHASES      = 8           # bare "temporal_phases"
DEFAULT_STRIDED_STRIDE       = 9973        # bare "strided_sweep"
DEFAULT_SAWTOOTH_PERIOD      = 0           # bare "sawtooth" => period == num_bins

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
        return (i * _U64(6364136223846793005)
                + _U64(1442695040888963407)
                + _U64(seed & _MASK64) * _U64(0x9E3779B97F4A7C15))


def scatter_bin(rank: int, num_bins: int, offset: int) -> int:
    """Port of scatter_bin: (rank*kScatterPrime + offset) % num_bins, in uint64."""
    return int((( (rank & _MASK64) * K_SCATTER_PRIME + offset) & _MASK64) % num_bins)


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
    bits = int(n - 1).bit_length()          # ceil(log2(n)) for n >= 2
    half = (bits + 1) // 2
    mask = _MASK64 if half >= 64 else ((1 << half) - 1)
    mask = _U64(mask)
    hh = _U64(half)
    x = i.copy()
    todo = np.ones(x.shape, dtype=bool)
    with np.errstate(over="ignore"):
        for _ in range(128):                # cycle-walk; expected < 4 iters
            xs = x[todo]
            l = (xs >> hh) & mask
            r = xs & mask
            for rnd in range(4):
                nl = r
                nr = l ^ (_feistel_mix(r, seed + rnd) & mask)
                l, r = nl, nr
            x[todo] = (l << hh) | r
            todo = x >= _U64(n)
            if not todo.any():
                break
    return x


def _scatter_bin_vec(ranks: np.ndarray, num_bins: int, offset: int) -> np.ndarray:
    r = ranks.astype(object)  # python big-ints to avoid overflow before mod
    return np.array([(int(x) * K_SCATTER_PRIME + offset) % num_bins for x in r], dtype=np.int64)


# ---------------------------------------------------------------------------
# Spec + parse — mirror ShapeSpec / parse_input_shape / knob_or.
# ---------------------------------------------------------------------------

VALID_SHAPES = {
    "concentrated", "powerlaw", "zipf", "hash_synonym",
    "stale_resident", "temporal_phases", "strided_sweep",
    "sawtooth",
}


@dataclass
class ShapeSpec:
    shape: str
    knob: float = 0.0
    has_knob: bool = False


def parse_input_shape(spec: str) -> ShapeSpec:
    """Port of parse_input_shape: "name" or "name:knob"."""
    name = spec
    knob = 0.0
    has_knob = False
    if ":" in spec:
        name, knob_s = spec.split(":", 1)
        knob = float(knob_s)
        has_knob = True
    if name not in VALID_SHAPES:
        raise ValueError(f"Unknown InputShape: {spec}")
    return ShapeSpec(shape=name, knob=knob, has_knob=has_knob)


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

def build_pmf(spec: ShapeSpec, num_bins: int, seed: int) -> np.ndarray:
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

    elif spec.shape == "zipf":
        s = knob_or(spec, DEFAULT_ZIPF_EXPONENT)
        w = ranked_powerlaw(num_bins, s)
        idx = _scatter_bin_vec(np.arange(num_bins), num_bins, offset)
        pmf[idx] = w

    elif spec.shape == "hash_synonym":
        hot_share = knob_or(spec, DEFAULT_HASH_SYNONYM_HOTSHARE)
        slot = offset % K_ADVERSARIAL_CACHE_SLOTS
        syn = [slot + k * K_ADVERSARIAL_CACHE_SLOTS
               for k in range(K_HASH_SYNONYM_COUNT)
               if slot + k * K_ADVERSARIAL_CACHE_SLOTS < num_bins]
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
    return shape in ("stale_resident", "temporal_phases", "strided_sweep", "sawtooth")


# ---------------------------------------------------------------------------
# Core generation — mirror generate_shape_impl. Returns BIN INDICES (the value
# mapping is applied by the public entry points below).
# ---------------------------------------------------------------------------

def _bins_for_spec(spec: ShapeSpec, n: int, num_bins: int, seed: int) -> np.ndarray:
    offset = seed % num_bins

    # Exact-uniform endpoint: exact round-robin counts, but in pseudo-random
    # sequence order (a Feistel permutation of the tiling). Uniform counts,
    # randomly distributed in the input -- NOT the sequential ramp (= sawtooth).
    if spec.shape == "concentrated" and spec.has_knob and spec.knob >= 1.0:
        perm = permute_index(np.arange(n, dtype=_U64), n, seed)
        return (perm % np.uint64(num_bins)).astype(np.int64)

    if not is_ordering_shape(spec.shape):
        pmf = build_pmf(spec, num_bins, seed)
        cdf = np.cumsum(pmf)
        cdf[-1] = 1.0  # guard fp drift, matches header
        u = u01_from_hash(element_key(np.arange(n, dtype=_U64), seed))
        bins = np.searchsorted(cdf, u, side="right").astype(np.int64)  # == upper_bound_cdf
        np.clip(bins, 0, num_bins - 1, out=bins)
        return bins

    if spec.shape == "strided_sweep":
        stride = int(round(spec.knob)) if spec.has_knob else DEFAULT_STRIDED_STRIDE
        i = np.arange(n, dtype=object)  # big-int to mirror uint64 multiply then mod
        return np.array([(int(x) * stride) % num_bins for x in i], dtype=np.int64)

    if spec.shape == "sawtooth":
        # bin = i % period; period 0 (the default) means a full num_bins sweep.
        requested = int(round(spec.knob)) if spec.has_knob else DEFAULT_SAWTOOTH_PERIOD
        period = num_bins if requested == 0 else min(requested, num_bins)
        return (np.arange(n, dtype=np.int64) % period)

    if spec.shape == "temporal_phases":
        requested = int(round(spec.knob)) if spec.has_knob else DEFAULT_TEMPORAL_PHASES
        phases = max(1, min(requested, num_bins))
        i = np.arange(n, dtype=np.int64)
        phase = (i * phases) // n
        np.clip(phase, 0, phases - 1, out=phase)
        step = num_bins // phases + 1
        return _scatter_bin_vec(phase.astype(object) * step, num_bins, offset)

    if spec.shape == "stale_resident":
        # A cold working set of `span` distinct bins, swept cyclically (k = i*odd
        # % span) and scattered across the array. Recurs in every block but
        # overflows the per-block cache when span > slots -> thrashes it.
        cover = knob_or(spec, 2.0)
        want = int(round(cover * K_ADVERSARIAL_CACHE_SLOTS))
        span = max(1, min(want, num_bins))
        k = (np.arange(n, dtype=_U64) * _U64(2654435761)) % _U64(span)
        return _scatter_bin_vec(k.astype(object), num_bins, offset)

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

def generate_histogram_input_even(spec, n, num_bins, lower, upper, dtype=np.int32, seed=42):
    if isinstance(spec, str):
        spec = parse_input_shape(spec)
    bins = _bins_for_spec(spec, n, num_bins, seed)
    values = _even_bin_to_value(bins, num_bins, lower, upper, dtype)
    return values, bins


def generate_histogram_input_range(spec, n, num_bins, levels, dtype=np.int32, seed=42):
    if isinstance(spec, str):
        spec = parse_input_shape(spec)
    levels = np.asarray(levels)
    bins = _bins_for_spec(spec, n, num_bins, seed)
    values = _range_bin_to_value(bins, levels, dtype)
    return values, bins


# Convenience for the viz scripts: bins only (EVEN-equivalent), no value map.
def generate_bins(spec, n, num_bins, seed=42):
    if isinstance(spec, str):
        spec = parse_input_shape(spec)
    return _bins_for_spec(spec, n, num_bins, seed)


# ---------------------------------------------------------------------------
# Self-test: confirm the mirror reproduces the documented behavior.
# ---------------------------------------------------------------------------

def _demo():
    N, B = 200_000, 64

    print("== exact endpoints ==")
    b = generate_bins("concentrated:1.0", N, B, seed=42)
    c = np.bincount(b, minlength=B)
    print(f"  concentrated:1.0  per-bin min={c.min()} max={c.max()} (exact uniform), H={normalized_entropy(c/N):.4f}")
    b = generate_bins("concentrated:0.0", N, B, seed=42)
    c = np.bincount(b, minlength=B)
    print(f"  concentrated:0.0  nonzero bins={np.count_nonzero(c)} at bin {c.argmax()} (constant, scattered off 0)")

    print("== concentrated entropy knob (measured top-bin share) ==")
    for e in (1.0, 0.75, 0.5, 0.25, 0.0):
        b = generate_bins(f"concentrated:{e}", N, B, seed=42)
        c = np.bincount(b, minlength=B)
        print(f"  E={e:.2f}: H={normalized_entropy(c/N):.3f}  top-share={c.max()/N:.1%}")

    print("== hot bin varies with seed (concentrated:0.3) ==")
    args = {int(np.bincount(generate_bins('concentrated:0.3', N, B, seed=s), minlength=B).argmax()) for s in range(1, 6)}
    print(f"  argmax bins across seeds 1..5: {sorted(args)} (not pinned to 0)")

    print("== adversarial: hash_synonym collides on one slot ==")
    b = generate_bins("hash_synonym", N, 262144, seed=42)
    hotbins = np.argsort(np.bincount(b, minlength=262144))[::-1][:K_HASH_SYNONYM_COUNT]
    print(f"  distinct slots among the {K_HASH_SYNONYM_COUNT} hottest bins: {len(set(int(x) % 4096 for x in hotbins))} (want 1)")

    print("== adversarial: temporal_phases moves the hot bin ==")
    b = generate_bins("temporal_phases:4", N, 256, seed=42)
    q = N // 4
    print(f"  per-quarter hot bin: {[int(np.bincount(b[i*q:(i+1)*q], minlength=256).argmax()) for i in range(4)]}")


if __name__ == "__main__":
    _demo()
