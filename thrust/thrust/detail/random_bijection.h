/*
 *  Copyright 2008-2025 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

//! \file random_bijection.h
//! \brief An implementation of a bijective function for use in shuffling

#pragma once

#include <thrust/detail/config.h>

#include <thrust/random.h>

#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN
namespace detail
{
//! \brief A Feistel cipher for operating on power of two sized problems
class feistel_bijection
{
  struct round_state
  {
    std::uint32_t left;
    std::uint32_t right;
  };

public:
  using index_type = std::uint64_t;

  template <class URBG>
  _CCCL_HOST_DEVICE feistel_bijection(std::uint64_t m, URBG&& g)
  {
    std::uint64_t total_bits = get_cipher_bits(m);
    // Half bits rounded down
    left_side_bits = total_bits / 2;
    left_side_mask = (1ull << left_side_bits) - 1;
    // Half the bits rounded up
    right_side_bits = total_bits - left_side_bits;
    right_side_mask = (1ull << right_side_bits) - 1;

    thrust::uniform_int_distribution<std::uint32_t> dist;
    for (std::uint32_t i = 0; i < num_rounds; i++)
    {
      key[i] = dist(g);
    }
  }

  _CCCL_HOST_DEVICE std::uint64_t nearest_power_of_two() const
  {
    return 1ull << (left_side_bits + right_side_bits);
  }

  _CCCL_HOST_DEVICE std::uint64_t size() const
  {
    return nearest_power_of_two();
  }

  _CCCL_HOST_DEVICE std::uint64_t operator()(const std::uint64_t val) const
  {
    std::uint32_t state[2] = {static_cast<std::uint32_t>(val >> right_side_bits),
                              static_cast<std::uint32_t>(val & right_side_mask)};
    for (std::uint32_t i = 0; i < num_rounds; i++)
    {
      std::uint32_t hi, lo;
      constexpr std::uint64_t M0 = UINT64_C(0xD2B74407B1CE6E93);
      mulhilo(M0, state[0], hi, lo);
      lo       = (lo << (right_side_bits - left_side_bits)) | state[1] >> left_side_bits;
      state[0] = ((hi ^ key[i]) ^ state[1]) & left_side_mask;
      state[1] = lo & right_side_mask;
    }
    // Combine the left and right sides together to get result
    return (static_cast<std::uint64_t>(state[0]) << right_side_bits) | static_cast<std::uint64_t>(state[1]);
  }

private:
  // Perform 64 bit multiplication and save result in two 32 bit int
  static _CCCL_HOST_DEVICE void mulhilo(std::uint64_t a, std::uint64_t b, std::uint32_t& hi, std::uint32_t& lo)
  {
    std::uint64_t product = a * b;
    hi                    = static_cast<std::uint32_t>(product >> 32);
    lo                    = static_cast<std::uint32_t>(product);
  }

  // Find the nearest power of two
  static _CCCL_HOST_DEVICE std::uint64_t get_cipher_bits(std::uint64_t m)
  {
    if (m <= 16)
    {
      return 4;
    }
    std::uint64_t i = 0;
    m--;
    while (m != 0)
    {
      i++;
      m >>= 1;
    }
    return i;
  }

  static constexpr std::uint32_t num_rounds = 24;
  std::uint64_t right_side_bits;
  std::uint64_t left_side_bits;
  std::uint64_t right_side_mask;
  std::uint64_t left_side_mask;
  std::uint32_t key[num_rounds];
};

//! \brief Adaptor for a bijection to work with any size problem. It achieves this by iterating the bijection until
//! the result is less than n. For feistel_bijection, the worst case number of iterations required for one call to
//! operator() is O(n) with low probability. It has amortised O(1) complexity.
//! \tparam IndexType The type of the index to shuffle.
//! \tparam Bijection The bijection to use. A low quality random bijection may lead to poor work balancing between calls
//! to the operator().
template <class IndexType, class Bijection = feistel_bijection>
class random_bijection
{
private:
  static_assert(::cuda::std::is_integral_v<IndexType>, "IndexType must be an integral type");
  static_assert(::cuda::std::is_convertible_v<IndexType, typename Bijection::index_type>,
                "IndexType must be convertible to Bijection::index_type");

  Bijection bijection;
  IndexType n;

public:
  using index_type = IndexType;

  template <class URBG>
  _CCCL_HOST_DEVICE random_bijection(IndexType n, URBG&& g)
      : bijection(n, ::cuda::std::forward<URBG>(g))
      , n(n)
  {}

  _CCCL_HOST_DEVICE IndexType operator()(IndexType i) const
  {
    auto upcast_i = static_cast<typename Bijection::index_type>(i);
    auto upcast_n = static_cast<typename Bijection::index_type>(n);

    // If i < n Iterating a bijection like this will always terminate.
    // If i >= n, then this may loop forever.
    if (upcast_i >= upcast_n)
    { // Avoid infinite loop.
      _CCCL_ASSERT(false, "index out of range");
      return upcast_i;
    }

    do
    {
      upcast_i = bijection(upcast_i);
    } while (upcast_i >= upcast_n);
    return static_cast<IndexType>(upcast_i);
  }

  _CCCL_HOST_DEVICE IndexType size() const
  {
    return n;
  }
};
} // namespace detail
THRUST_NAMESPACE_END
