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
#include <cuda/std/bit>
#include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN
namespace detail
{
//! \brief A Feistel cipher for operating on power of two sized problems
class feistel_bijection
{
public:
  using index_type = std::uint64_t;

  template <class URBG>
  _CCCL_HOST_DEVICE feistel_bijection(std::uint64_t m, URBG&& g)
  {
    // Calculate number of bits needed to represent num_elements - 1
    // Prevent zero
    const uint64_t max_index  = ::cuda::std::max(static_cast<uint64_t>(1), m) - 1;
    const uint64_t total_bits = static_cast<uint64_t>(::cuda::std::max(8, ::cuda::std::bit_width(max_index)));
    // Half bits rounded down
    L_bits = total_bits / 2;
    L_mask = (1ull << L_bits) - 1;
    // Half the bits rounded up
    R_bits = total_bits - L_bits;
    R_mask = (1ull << R_bits) - 1;

    thrust::uniform_int_distribution<std::uint32_t> dist;
    for (std::uint32_t i = 0; i < num_rounds; i++)
    {
      key[i] = dist(g);
    }
  }

  _CCCL_HOST_DEVICE std::uint64_t size() const
  {
    return 1ull << (L_bits + R_bits);
  }

  _CCCL_HOST_DEVICE std::uint64_t operator()(const std::uint64_t val) const
  {
    // Unfortunately this is duplicated with libcudacxx/include/cuda/__random/feistel_bijection.h
    // We cannot use the above because thrust PRNG generators incorrectly implement URBG requirements.
    // Mitchell, Rory, et al. "Bandwidth-optimal random shuffling for GPUs." ACM Transactions on Parallel Computing 9.1
    // (2022): 1-20.
    uint32_t L = static_cast<uint32_t>(val >> R_bits);
    uint32_t R = static_cast<uint32_t>(val & R_mask);
    for (uint32_t i = 0; i < num_rounds; i++)
    {
      constexpr uint64_t m0  = 0xD2B74407B1CE6E93;
      const uint64_t product = m0 * L;
      uint32_t F_k           = (product >> 32) ^ key[i];
      uint32_t B_k           = static_cast<uint32_t>(product);
      uint32_t L_prime       = F_k ^ R;

      uint32_t R_prime = (B_k << (R_bits - L_bits)) | R >> L_bits;
      L                = L_prime & L_mask;
      R                = R_prime & R_mask;
    }
    // Combine the left and right sides together to get result
    return (static_cast<uint64_t>(L) << R_bits) | static_cast<uint64_t>(R);
  }

private:
  static constexpr std::uint32_t num_rounds = 24;
  std::uint64_t R_bits;
  std::uint64_t L_bits;
  std::uint64_t R_mask;
  std::uint64_t L_mask;
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
