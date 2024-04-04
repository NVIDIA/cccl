/*******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <cuda/std/bit>
#include <cuda/std/type_traits>

#include <algorithm>

#include "cub/thread/thread_operators.cuh"
#include "cub/thread/thread_sort.cuh"
#include "test_util.h"
#include "test_util_sort.h"

/**
 * Population-count comparison operator.
 *
 * This operator can only be used with unsigned integral keys.
 * It allows to test:
 *  (a) A non-trivial comparison operator.
 *  (b) Stability, because many different keys have the same population count.
 */
struct PopcountLess
{
  template <typename T>
  __host__ __device__ inline bool operator()(const T& lhs, const T& rhs) const
  {
    return cuda::std::popcount(lhs) < cuda::std::popcount(rhs);
  }
};

template <typename KeyT, typename ValueT, typename CompareOp, int ItemsPerThread, bool IsStable>
__global__ void kernel(const KeyT* keys_in, KeyT* keys_out, const ValueT* values_in, ValueT* values_out)
{
  constexpr bool KEYS_ONLY = ::cuda::std::is_same<ValueT, cub::NullType>::value;

  KeyT thread_keys[ItemsPerThread];
  ValueT thread_values[ItemsPerThread];

  const auto thread_offset = ItemsPerThread * threadIdx.x;
  keys_in += thread_offset;
  keys_out += thread_offset;
  values_in += thread_offset;
  values_out += thread_offset;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    thread_keys[item] = keys_in[item];
    if (!KEYS_ONLY)
    {
      thread_values[item] = values_in[item];
    }
  }

  cub::ThreadNetworkSort<KeyT,
                         ValueT,
                         CompareOp,
                         ItemsPerThread,
                         IsStable ? cub::stability_t::stable : cub::stability_t::unstable>(
    thread_keys, thread_values, CompareOp{});

  for (int item = 0; item < ItemsPerThread; item++)
  {
    keys_out[item] = thread_keys[item];
    if (!KEYS_ONLY)
    {
      values_out[item] = thread_values[item];
    }
  }
}

template <typename KeyT, typename ValueT, typename CompareOp, int ItemsPerThread, bool IsStable>
void Test()
{
  constexpr bool KEYS_ONLY                = ::cuda::std::is_same<ValueT, cub::NullType>::value;
  constexpr unsigned int threads_in_block = 1024;
  constexpr unsigned int elements         = threads_in_block * ItemsPerThread;

  using MaskedValueT = cub::detail::conditional_t< std::is_same<ValueT, cub::NullType>::value, KeyT, ValueT>;

  thrust::default_random_engine re;
  thrust::device_vector<std::uint8_t> data_source(elements);

  for (int iteration = 0; iteration < 10; iteration++)
  {
    thrust::sequence(data_source.begin(), data_source.end());
    thrust::shuffle(data_source.begin(), data_source.end(), re);
    thrust::device_vector<KeyT> in_keys(data_source);
    thrust::device_vector<KeyT> out_keys(elements);

    thrust::shuffle(data_source.begin(), data_source.end(), re);
    thrust::device_vector<MaskedValueT> in_values(data_source);
    thrust::device_vector<MaskedValueT> out_values(elements);

    thrust::host_vector<KeyT> host_keys(in_keys);
    thrust::host_vector<MaskedValueT> host_values(in_values);

    if (KEYS_ONLY)
    {
      kernel<KeyT, ValueT, CompareOp, ItemsPerThread, IsStable><<<1, threads_in_block>>>(
        thrust::raw_pointer_cast(in_keys.data()),
        thrust::raw_pointer_cast(out_keys.data()),
        (ValueT*) nullptr,
        (ValueT*) nullptr);
    }
    else
    {
      kernel<KeyT, MaskedValueT, CompareOp, ItemsPerThread, IsStable><<<1, threads_in_block>>>(
        thrust::raw_pointer_cast(in_keys.data()),
        thrust::raw_pointer_cast(out_keys.data()),
        thrust::raw_pointer_cast(in_values.data()),
        thrust::raw_pointer_cast(out_values.data()));
    }

    if (IsStable)
    {
      // If the sort is stable, we expect perfect equality of the keys and values
      // with a reference stable sort such as the C++ standard library.

      for (unsigned int tid = 0; tid < threads_in_block; tid++)
      {
        const auto thread_begin = tid * ItemsPerThread;
        const auto thread_end   = thread_begin + ItemsPerThread;

        if (KEYS_ONLY)
        {
          std::stable_sort(thrust::raw_pointer_cast(host_keys.data() + thread_begin),
                           thrust::raw_pointer_cast(host_keys.data() + thread_end),
                           CompareOp{});
        }
        else
        {
          // Pack pairs.
          std::vector<std::pair<KeyT, MaskedValueT>> h_pairs(ItemsPerThread);
          for (int item_idx = 0; item_idx < ItemsPerThread; item_idx++)
          {
            h_pairs[item_idx] =
              std::make_pair(host_keys[thread_begin + item_idx], host_values[thread_begin + item_idx]);
          }

          std::stable_sort(
            h_pairs.begin(),
            h_pairs.end(),
            [](const std::pair<KeyT, MaskedValueT>& lhs, const std::pair<KeyT, MaskedValueT>& rhs) -> bool {
              CompareOp compare_op;
              return compare_op(lhs.first, rhs.first);
            });

          // Unpack pairs.
          for (int item_idx = 0; item_idx < ItemsPerThread; item_idx++)
          {
            host_keys[thread_begin + item_idx]   = h_pairs[item_idx].first;
            host_values[thread_begin + item_idx] = h_pairs[item_idx].second;
          }
        }
      }

      AssertEquals(host_keys, out_keys);
      if (!KEYS_ONLY)
      {
        AssertEquals(host_values, out_values);
      }
    }
    else
    {
      // If the sort is unstable, there are multiple valid outputs.
      // We check that the output is a permutation of the input and that it is sorted.

      thrust::host_vector<KeyT> host_out_keys(out_keys);
      thrust::host_vector<MaskedValueT> host_out_values(out_values);

      for (unsigned int tid = 0; tid < threads_in_block; tid++)
      {
        const auto thread_begin = tid * ItemsPerThread;

        AssertTrue(IsSorted(
          host_out_keys.begin() + thread_begin, host_out_keys.begin() + thread_begin + ItemsPerThread, CompareOp{}));
        if (KEYS_ONLY)
        {
          AssertTrue(IsPermutationOf(thrust::raw_pointer_cast(host_out_keys.data() + thread_begin),
                                     thrust::raw_pointer_cast(host_keys.data() + thread_begin),
                                     ItemsPerThread));
        }
        else
        {
          AssertTrue(IsPermutationOf(
            thrust::raw_pointer_cast(host_out_keys.data() + thread_begin),
            thrust::raw_pointer_cast(host_keys.data() + thread_begin),
            thrust::raw_pointer_cast(host_out_values.data() + thread_begin),
            thrust::raw_pointer_cast(host_values.data() + thread_begin),
            ItemsPerThread));
        }
      }
    }
  }
}

template <typename KeyT, typename ValueT, typename CompareOp, bool IsStable>
void Test()
{
  Test<KeyT, ValueT, CompareOp, 2, IsStable>();
  Test<KeyT, ValueT, CompareOp, 3, IsStable>();
  Test<KeyT, ValueT, CompareOp, 4, IsStable>();
  Test<KeyT, ValueT, CompareOp, 5, IsStable>();
  Test<KeyT, ValueT, CompareOp, 7, IsStable>();
  Test<KeyT, ValueT, CompareOp, 8, IsStable>();
  Test<KeyT, ValueT, CompareOp, 9, IsStable>();
  Test<KeyT, ValueT, CompareOp, 11, IsStable>();
}

int main()
{
  Test<std::uint32_t, std::uint32_t, PopcountLess, true>();
  Test<std::uint32_t, std::uint32_t, PopcountLess, false>();
  Test<std::uint32_t, std::uint64_t, PopcountLess, true>();
  Test<std::uint32_t, std::uint64_t, PopcountLess, false>();
  Test<std::uint64_t, std::uint32_t, PopcountLess, true>();
  Test<std::uint64_t, std::uint32_t, PopcountLess, false>();
  Test<std::uint64_t, std::uint64_t, PopcountLess, true>();
  Test<std::uint64_t, std::uint64_t, PopcountLess, false>();
  Test<std::uint32_t, cub::NullType, PopcountLess, true>();
  Test<std::uint32_t, cub::NullType, PopcountLess, false>();
  Test<std::int32_t, cub::NullType, cub::Less, true>();
  Test<std::int32_t, cub::NullType, cub::Less, false>();
  Test<std::int32_t, cub::NullType, cub::Greater, true>();
  Test<std::int32_t, cub::NullType, cub::Greater, false>();
  Test<float, cub::NullType, cub::Less, true>();
  Test<float, cub::NullType, cub::Less, false>();
  Test<float, cub::NullType, cub::Greater, true>();
  Test<float, cub::NullType, cub::Greater, false>();

  return 0;
}
