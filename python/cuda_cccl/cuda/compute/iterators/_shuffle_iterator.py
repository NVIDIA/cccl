# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import ctypes

from numba import cuda, int64, types, uint32, uint64

from .._caching import cache_with_key
from ._iterators import IteratorBase, IteratorKind

# Number of Feistel rounds (matches C++ __feistel_bijection)
NUM_ROUNDS = 24

# Feistel multiplier (same as C++)
FEISTEL_M0 = 0xD2B74407B1CE6E93

# SplitMix64 constants for key derivation
SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX64_MUL1 = 0xBF58476D1CE4E5B9
SPLITMIX64_MUL2 = 0x94D049BB133111EB


@cuda.jit(device=True, inline=True)
def _splitmix64_next(state):
    """
    Device-side SplitMix64 step. Returns (next_state, output).
    Used to generate independent keys from a seed.
    """
    GAMMA = uint64(SPLITMIX64_GAMMA)
    MUL1 = uint64(SPLITMIX64_MUL1)
    MUL2 = uint64(SPLITMIX64_MUL2)

    state = state + GAMMA
    z = state
    z = (z ^ (z >> uint64(30))) * MUL1
    z = (z ^ (z >> uint64(27))) * MUL2
    z = z ^ (z >> uint64(31))
    return state, z


@cuda.jit(device=True, inline=True)
def _feistel_bijection(val, seed, left_bits, right_bits, left_mask, right_mask):
    """
    Feistel bijection matching libcudacxx __feistel_bijection.
    """
    M0 = uint64(FEISTEL_M0)

    # Match C++ initialization exactly:
    #   __state.__low  = val >> right_side_bits
    #   __state.__high = val & right_side_mask
    state_low = uint32((val >> uint64(right_bits)) & uint64(left_mask))
    state_high = uint32(val & uint64(right_mask))

    shift_amount = uint64(right_bits - left_bits)
    lbits = uint64(left_bits)
    lmask = uint32(left_mask)
    rmask = uint32(right_mask)

    # Initialize key generator state from seed
    key_state = uint64(seed)

    # 24 rounds with independent keys
    for _ in range(NUM_ROUNDS):
        # Generate next key using SplitMix64
        key_state, key_output = _splitmix64_next(key_state)
        round_key = uint32(key_output & uint64(0xFFFFFFFF))

        # Feistel round matching C++ exactly:
        #   product = M0 * __state.__high
        #   hi = product >> 32
        #   lo = product & 0xFFFFFFFF
        #   lo = (lo << shift) | (__state.__low >> left_bits)
        #   __state.__high = (hi ^ key ^ __state.__low) & left_mask
        #   __state.__low = lo & right_mask
        product = M0 * uint64(state_high)
        hi = uint32(product >> uint64(32))
        lo = uint32(product)

        lo = uint32((uint64(lo) << shift_amount) | (uint64(state_low) >> lbits))

        new_high = ((hi ^ round_key) ^ state_low) & lmask
        new_low = lo & rmask

        state_high = new_high
        state_low = new_low

    # Match C++ output: (__state.__high << right_bits) | __state.__low
    return (uint64(state_high) << uint64(right_bits)) | uint64(state_low)


def _splitmix64_host(x: int) -> int:
    """
    Host-side SplitMix64 used to derive a 64-bit seed from the user seed.
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


class ShuffleIteratorKind(IteratorKind):
    pass


# Cache key excludes seed - only structure-defining parameters
def _make_cache_key(num_items: int, seed: int):
    return (num_items,)


@cache_with_key(_make_cache_key)
def _make_shuffle_iterator_class(num_items: int, seed: int):
    """
    Factory that creates a ShuffleIterator class for a given num_items.
    The seed is NOT part of the cache key, so the same class is reused for different seeds.
    """
    if num_items <= 0:
        raise ValueError("num_items must be > 0")

    m = int(num_items)

    # total_bits = ceil(log2(m)), minimum 4 bits for proper mixing (matches C++)
    total_bits = max((m - 1).bit_length(), 4)

    # Feistel uses unbalanced halves: left = floor(total/2), right = ceil(total/2)
    left_bits = total_bits // 2
    right_bits = total_bits - left_bits

    if total_bits > 63:
        raise ValueError("num_items too large for uint64-based shuffle iterator")

    left_mask = (1 << left_bits) - 1
    right_mask = (1 << right_bits) - 1

    # Capture constants for the device functions
    _m = m
    _left_bits = left_bits
    _right_bits = right_bits
    _left_mask = left_mask
    _right_mask = right_mask

    @cuda.jit(device=True)
    def _permute_with_seed(index, seed):
        """Permute a single index using the Feistel bijection with cycle-walking."""
        mm = uint64(_m)
        x = uint64(index)

        y = _feistel_bijection(
            x,
            seed,
            _left_bits,
            _right_bits,
            uint64(_left_mask),
            uint64(_right_mask),
        )

        # Cycle-walk into [0, m)
        while y >= mm:
            y = _feistel_bijection(
                y,
                seed,
                _left_bits,
                _right_bits,
                uint64(_left_mask),
                uint64(_right_mask),
            )

        return int64(y)

    # State: (index, seed) - matches C++ which stores (bijection, current_index)
    state_type = types.UniTuple(types.int64, 2)

    class ShuffleIterator(IteratorBase):
        iterator_kind_type = ShuffleIteratorKind

        def __init__(self, seed: int):
            # State: (current_index, seed)
            # One iterator = one permutation (matches C++ behavior)
            cvalue = (ctypes.c_int64 * 2)(0, seed)
            super().__init__(
                cvalue=cvalue,
                state_type=state_type,
                value_type=types.int64,
            )

        @property
        def host_advance(self):
            return ShuffleIterator._advance

        @property
        def advance(self):
            return ShuffleIterator._advance

        @property
        def input_dereference(self):
            return ShuffleIterator._input_dereference

        @property
        def output_dereference(self):
            raise AttributeError("ShuffleIterator cannot be used as an output iterator")

        @staticmethod
        def _advance(state, distance):
            idx = state[0][0]
            seed = state[0][1]
            state[0] = (idx + distance, seed)

        @staticmethod
        def _input_dereference(state, result):
            idx = state[0][0]
            seed = state[0][1]
            result[0] = _permute_with_seed(idx, seed)

    return ShuffleIterator


def make_shuffle_iterator(num_items: int, seed: int):
    """
    Iterator that produces a deterministic "random" permutation
    of indices in ``[0, num_items)``.

    Uses a Feistel cipher bijection matching the libcudacxx implementation,
    with 24 rounds and independent keys per round for high-quality shuffling.

    Parameters
    ----------
    num_items : int
        Number of elements in the domain to permute.
    seed : int
        Seed used to parameterize the permutation. Different seeds produce
        different (deterministic) permutations.

    Returns
    -------
    ShuffleIterator
        An iterator that yields a shuffled ordering of indices in
        ``[0, num_items)``.
    """
    # Get the class (cached by num_items only, NOT seed)
    ShuffleIteratorClass = _make_shuffle_iterator_class(num_items, seed)

    # Derive the internal seed from the user seed
    internal_seed = _splitmix64_host(int(seed))

    # Create instance with the runtime seed
    return ShuffleIteratorClass(internal_seed)
