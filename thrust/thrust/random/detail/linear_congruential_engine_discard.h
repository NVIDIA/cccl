// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/random/detail/mod.h>

#include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN

namespace random::detail
{
template <typename UIntType, UIntType A, unsigned long long C, UIntType M>
struct linear_congruential_engine_discard_implementation
{
  _CCCL_HOST_DEVICE static void discard(UIntType& state, unsigned long long z)
  {
    for (; z > 0; --z)
    {
      state = detail::mod<UIntType, A, C, M>(state);
    }
  }
}; // end linear_congruential_engine_discard

// specialize for small integers and c == 0
// XXX figure out a robust implementation of this for any unsigned integer type later
template <std::uint32_t A, std::uint32_t M>
struct linear_congruential_engine_discard_implementation<std::uint32_t, A, 0, M>
{
  _CCCL_HOST_DEVICE static void discard(std::uint32_t& state, unsigned long long z)
  {
    const std::uint32_t modulus = M;

    // XXX we need to use unsigned long long here or we will encounter overflow in the
    //     multiplies below
    //     figure out a robust implementation of this later
    unsigned long long multiplier      = A;
    unsigned long long multiplier_to_z = 1;

    // see http://en.wikipedia.org/wiki/Modular_exponentiation
    while (z > 0)
    {
      if (z & 1)
      {
        // multiply in this bit's contribution while using modulus to keep result small
        multiplier_to_z = (multiplier_to_z * multiplier) % modulus;
      }

      // move to the next bit of the exponent, square (and mod) the base accordingly
      z >>= 1;
      multiplier = (multiplier * multiplier) % modulus;
    }

    state = static_cast<std::uint32_t>((multiplier_to_z * state) % modulus);
  }
}; // end linear_congruential_engine_discard

struct linear_congruential_engine_discard
{
  template <typename LinearCongruentialEngine>
  _CCCL_HOST_DEVICE static void discard(LinearCongruentialEngine& lcg, unsigned long long z)
  {
    using result_type                    = typename LinearCongruentialEngine::result_type;
    [[maybe_unused]] const result_type c = LinearCongruentialEngine::increment;
    [[maybe_unused]] const result_type a = LinearCongruentialEngine::multiplier;
    [[maybe_unused]] const result_type m = LinearCongruentialEngine::modulus;

    linear_congruential_engine_discard_implementation<result_type, a, c, m>::discard(lcg.m_x, z);
  }
}; // end linear_congruential_engine_discard
} // namespace random::detail

THRUST_NAMESPACE_END
