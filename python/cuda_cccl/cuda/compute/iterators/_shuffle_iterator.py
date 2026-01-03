# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from numba import cuda, int64, uint64

from .._caching import cache_with_key
from ._iterators import (
    CountingIterator as _CountingIterator,
)
from ._iterators import (
    make_transform_iterator,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# SplitMix64 step (≈ 2^64 / φ)
SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15

# SplitMix64 avalanche multipliers
SPLITMIX64_MUL1 = 0xBF58476D1CE4E5B9
SPLITMIX64_MUL2 = 0x94D049BB133111EB

# Per-round constant to decorrelate Feistel rounds (any odd 64-bit constant works)
FEISTEL_ROUND_C = 0xD6E8FEB86659FD93


@cuda.jit(device=True, inline=True)
def _mix64(z):
    """
    SplitMix64-style 64-bit mixing function.

    Used as the round function core inside the Feistel network.
    """
    z = uint64(z)
    z ^= z >> uint64(30)
    z = uint64(z * uint64(SPLITMIX64_MUL1))
    z ^= z >> uint64(27)
    z = uint64(z * uint64(SPLITMIX64_MUL2))
    z ^= z >> uint64(31)
    return z


@cuda.jit(device=True, inline=True)
def _feistel_balanced(x, key, half_bits, half_mask, rounds):
    """
    Balanced Feistel permutation over 2 * half_bits bits.

    The input domain is [0, 2^(2*half_bits)).
    This function defines a bijection on that domain.
    """
    hb = uint64(half_bits)

    # Split x into equal-width halves
    L = x & half_mask
    R = (x >> hb) & half_mask

    for rnd in range(rounds):
        # Round function F(R) -> half_bits bits
        z = R ^ key ^ uint64(rnd * FEISTEL_ROUND_C)
        F = _mix64(z) & half_mask

        # Standard Feistel step
        new_L = R
        new_R = (L ^ F) & half_mask
        L = new_L
        R = new_R

    return (R << hb) | L


def _splitmix64_host(x: int) -> int:
    """
    Host-side SplitMix64 used to derive a 64-bit key from the seed.
    """
    x &= (1 << 64) - 1
    x = (x + SPLITMIX64_GAMMA) & ((1 << 64) - 1)
    z = x
    z ^= z >> 30
    z = (z * SPLITMIX64_MUL1) & ((1 << 64) - 1)
    z ^= z >> 27
    z = (z * SPLITMIX64_MUL2) & ((1 << 64) - 1)
    z ^= z >> 31
    return z & ((1 << 64) - 1)


def _make_cache_key(num_items: int, seed: int, rounds: int):
    return (num_items, seed, rounds)


@cache_with_key(_make_cache_key)
def make_shuffle_iterator(num_items: int, seed: int, rounds: int = 8):
    """
    Iterator that produces a deterministic "random" permutation
    of indices in ``[0, num_items)``.



    Parameters
    ----------
    num_items : int
        Number of elements in the domain to permute.
    seed : int
        Seed used to parameterize the permutation. Different seeds produce
        different (deterministic) permutations.
    rounds : int, optional
        Number of Feistel rounds to use. More rounds improve diffusion at the
        cost of additional arithmetic. Typical values are 6–10.

    Returns
    -------
    TransformIterator
        An iterator that yields a shuffled ordering of indices in
        ``[0, num_items)``.


    Notes
    -----
    This iterator does **not** materialize a permutation table. Instead, it
    computes each permuted index on demand using a *stateless bijection* derived
    from a fixed seed.

    The iterator is implemented as::

        TransformIterator(CountingIterator(0), permute)

    where ``permute(i)`` is a pure function that maps ``i`` to a unique value in
    ``[0, num_items)``.

    The permutation is constructed as follows:

    1. Let ``k = ceil(log2(num_items))`` and ``h = ceil(k / 2)``.
       We define a permutation over ``2^(2h)`` elements (a power-of-two domain
       large enough to cover ``[0, num_items)``).

    2. A **balanced Feistel network** with ``h``-bit halves is used to define a
       bijection over this ``2^(2h)`` domain. Each Feistel round applies a simple,
       fast mixing function (SplitMix64-style) keyed by ``seed`` and the round
       index.

    3. To restrict the permutation to ``[0, num_items)``, **cycle-walking** is
       used: the Feistel permutation is repeatedly applied until the result lies
       within ``[0, num_items)``. This preserves bijectivity on the restricted
       domain.

    Properties
    ----------
    - **Bijective on ``[0, num_items)``**: every index appears exactly once.
    - **Deterministic**: the same ``num_items`` and ``seed`` always produce the
      same ordering.
    - **Stateless**: no per-element or per-thread state is required.
    - **Lazy**: indices are computed on demand; no permutation buffer is stored.
    - **Device-friendly**: implemented using simple integer arithmetic and
      device-callable functions.

    Limitations
    -----------
    - The resulting permutation is *not* uniformly sampled from all
      ``num_items!`` permutations. It is drawn from a large, structured family
      of permutations induced by the Feistel construction.
    - Cycle-walking may apply the Feistel permutation more than once per element
      when ``num_items`` is far from a power of two, though the expected number
      of iterations is close to 1.
    """
    if num_items <= 0:
        raise ValueError("num_items must be > 0")

    if rounds < 6:
        rounds = 6

    m = int(num_items)

    # k = ceil(log2(m))
    k = (m - 1).bit_length()

    # balanced halves: total_bits = 2 * h >= k
    h = (k + 1) // 2
    total_bits = 2 * h

    if total_bits > 63:
        raise ValueError("num_items too large for uint64-based shuffle iterator")

    half_mask = (1 << h) - 1
    full_mask = (1 << total_bits) - 1

    key = _splitmix64_host(int(seed))

    # Closure capturing only constants; device-callable helpers do the work
    def permute(i):
        mm = uint64(m)
        x = uint64(i) & uint64(full_mask)

        y = _feistel_balanced(
            x,
            uint64(key),
            h,
            uint64(half_mask),
            rounds,
        )

        # Cycle-walk into [0, m)
        while y >= mm:
            y = _feistel_balanced(
                y,
                uint64(key),
                h,
                uint64(half_mask),
                rounds,
            )

        return int64(y)

    return make_transform_iterator(_CountingIterator(int64(0)), permute, "input")
