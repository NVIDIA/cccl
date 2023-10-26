/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cub/detail/cpp_compatibility.cuh>

#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "catch2_test_helper.h"


template <typename InputIteratorT,
          typename OutputIteratorT,
          typename ActionT,
          int ItemsPerThread,
          int ThreadsInBlock,
          int RadixBits,
          bool Memoize,
          cub::BlockScanAlgorithm Algorithm,
          cudaSharedMemConfig ShmemConfig>
__global__ void kernel(
    ActionT action, 
    InputIteratorT input, 
    OutputIteratorT output,
    int begin_bit,
    int end_bit,
    bool striped)
{
  using key_t = cub::detail::value_t<InputIteratorT>;
  using block_radix_sort_t = cub::BlockRadixSort<key_t,
                                                 ThreadsInBlock,
                                                 ItemsPerThread,
                                                 cub::NullType,
                                                 RadixBits,
                                                 Memoize,
                                                 Algorithm,
                                                 ShmemConfig>;

  using storage_t = typename block_radix_sort_t::TempStorage;

  __shared__ storage_t storage;

  key_t keys[ItemsPerThread];

  for (int i = 0; i < ItemsPerThread; i++)
  {
    keys[i] = input[threadIdx.x * ItemsPerThread + i];
  }

  block_radix_sort_t block_radix_sort(storage);

  if (striped)
  {
    action(block_radix_sort,
           keys,
           begin_bit,
           end_bit,
           cub::Int2Type<1>{});

    for (int i = 0; i < ItemsPerThread; i++)
    {
      output[threadIdx.x + ThreadsInBlock * i] = keys[i];
    }
  }
  else
  {
    action(block_radix_sort,
           keys,
           begin_bit,
           end_bit,
           cub::Int2Type<0>{});

    for (int i = 0; i < ItemsPerThread; i++)
    {
      output[threadIdx.x * ItemsPerThread + i] = keys[i];
    }
  }
}

template <int ItemsPerThread,
          int ThreadsInBlock,
          int RadixBits,
          bool Memoize,
          cub::BlockScanAlgorithm Algorithm,
          cudaSharedMemConfig ShmemConfig,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ActionT>
void block_radix_sort(
    ActionT action,
    InputIteratorT input,
    OutputIteratorT output,
    int begin_bit,
    int end_bit,
    bool striped)
{
  kernel<InputIteratorT,
         OutputIteratorT,
         ActionT,
         ItemsPerThread,
         ThreadsInBlock,
         RadixBits,
         Memoize,
         Algorithm,
         ShmemConfig>
    <<<1, ThreadsInBlock>>>(action, input, output, begin_bit, end_bit, striped);

  REQUIRE( cudaSuccess == cudaPeekAtLastError() );
  REQUIRE( cudaSuccess == cudaDeviceSynchronize() );
}

template <typename InputKeyIteratorT,
          typename InputValueIteratorT,
          typename OutputKeyIteratorT,
          typename OutputValueIteratorT,
          typename ActionT,
          int ItemsPerThread,
          int ThreadsInBlock,
          int RadixBits,
          bool Memoize,
          cub::BlockScanAlgorithm Algorithm,
          cudaSharedMemConfig ShmemConfig>
__global__ void kernel(
    ActionT action, 
    InputKeyIteratorT input_keys, 
    InputValueIteratorT input_values,
    OutputKeyIteratorT output_keys, 
    OutputValueIteratorT output_values,
    int begin_bit,
    int end_bit,
    bool striped)
{
  using key_t = cub::detail::value_t<InputKeyIteratorT>;
  using value_t = cub::detail::value_t<InputValueIteratorT>;
  using block_radix_sort_t = cub::BlockRadixSort<key_t,
                                                 ThreadsInBlock,
                                                 ItemsPerThread,
                                                 value_t,
                                                 RadixBits,
                                                 Memoize,
                                                 Algorithm,
                                                 ShmemConfig>;

  using storage_t = typename block_radix_sort_t::TempStorage;
  __shared__ storage_t storage;

  key_t keys[ItemsPerThread];
  value_t values[ItemsPerThread];

  for (int i = 0; i < ItemsPerThread; i++)
  {
    keys[i] = input_keys[threadIdx.x * ItemsPerThread + i];
    values[i] = input_values[threadIdx.x * ItemsPerThread + i];
  }

  block_radix_sort_t block_radix_sort(storage);

  if (striped)
  {
    action(block_radix_sort,
           keys,
           values,
           begin_bit,
           end_bit,
           cub::Int2Type<1>{});

    for (int i = 0; i < ItemsPerThread; i++)
    {
      output_keys[threadIdx.x + ThreadsInBlock * i] = keys[i];
      output_values[threadIdx.x + ThreadsInBlock * i] = values[i];
    }
  }
  else 
  {
    action(block_radix_sort,
           keys,
           values,
           begin_bit,
           end_bit,
           cub::Int2Type<0>{});

    for (int i = 0; i < ItemsPerThread; i++)
    {
      output_keys[threadIdx.x * ItemsPerThread + i] = keys[i];
      output_values[threadIdx.x * ItemsPerThread + i] = values[i];
    }
  }
}

template <int ItemsPerThread,
          int ThreadsInBlock,
          int RadixBits,
          bool Memoize,
          cub::BlockScanAlgorithm Algorithm,
          cudaSharedMemConfig ShmemConfig,
          typename InputKeyIteratorT,
          typename InputValueIteratorT,
          typename OutputKeyIteratorT,
          typename OutputValueIteratorT,
          typename ActionT>
void block_radix_sort(
    ActionT action,
    InputKeyIteratorT input_keys,
    InputValueIteratorT input_values,
    OutputKeyIteratorT output_keys,
    OutputValueIteratorT output_values,
    int begin_bit,
    int end_bit,
    bool striped)
{
  kernel<InputKeyIteratorT,
         InputValueIteratorT,
         OutputKeyIteratorT,
         OutputValueIteratorT,
         ActionT,
         ItemsPerThread,
         ThreadsInBlock,
         RadixBits,
         Memoize,
         Algorithm,
         ShmemConfig><<<1, ThreadsInBlock>>>(action,
                                             input_keys,
                                             input_values,
                                             output_keys,
                                             output_values,
                                             begin_bit,
                                             end_bit,
                                             striped);

  REQUIRE( cudaSuccess == cudaPeekAtLastError() );
  REQUIRE( cudaSuccess == cudaDeviceSynchronize() );
}

struct sort_op_t
{
  template <class BlockRadixSortT, class KeysT>
  __device__ void operator()(BlockRadixSortT &block_radix_sort,
                             KeysT &keys,
                             int begin_bit,
                             int end_bit,
                             cub::Int2Type<0> /* striped */)
  {
    block_radix_sort.Sort(keys, begin_bit, end_bit);
  }

  template <class BlockRadixSortT, class KeysT>
  __device__ void operator()(BlockRadixSortT &block_radix_sort,
                             KeysT &keys,
                             int begin_bit,
                             int end_bit,
                             cub::Int2Type<1> /* striped */)
  {
    block_radix_sort.SortBlockedToStriped(keys, begin_bit, end_bit);
  }
};

struct descending_sort_op_t
{
  template <class BlockRadixSortT, class KeysT>
  __device__ void operator()(BlockRadixSortT &block_radix_sort,
                             KeysT &keys,
                             int begin_bit,
                             int end_bit,
                             cub::Int2Type<0> /* striped */)
  {
    block_radix_sort.SortDescending(keys, begin_bit, end_bit);
  }

  template <class BlockRadixSortT, class KeysT>
  __device__ void operator()(BlockRadixSortT &block_radix_sort,
                             KeysT &keys,
                             int begin_bit,
                             int end_bit,
                             cub::Int2Type<1> /* striped */)
  {
    block_radix_sort.SortDescendingBlockedToStriped(keys, begin_bit, end_bit);
  }
};

struct sort_pairs_op_t
{
  template <class BlockRadixSortT, class KeysT, class ValuesT>
  __device__ void operator()(BlockRadixSortT &block_radix_sort,
                             KeysT &keys,
                             ValuesT &values,
                             int begin_bit,
                             int end_bit,
                             cub::Int2Type<0> /* striped */)
  {
    block_radix_sort.Sort(keys, values, begin_bit, end_bit);
  }

  template <class BlockRadixSortT, class KeysT, class ValuesT>
  __device__ void operator()(BlockRadixSortT &block_radix_sort,
                             KeysT &keys,
                             ValuesT &values,
                             int begin_bit,
                             int end_bit,
                             cub::Int2Type<1> /* striped */)
  {
    block_radix_sort.SortBlockedToStriped(keys, values, begin_bit, end_bit);
  }
};

struct descending_sort_pairs_op_t
{
  template <class BlockRadixSortT, class KeysT, class ValuesT>
  __device__ void operator()(BlockRadixSortT &block_radix_sort,
                             KeysT &keys,
                             ValuesT &values,
                             int begin_bit,
                             int end_bit,
                             cub::Int2Type<0> /* striped */)
  {
    block_radix_sort.SortDescending(keys, values, begin_bit, end_bit);
  }

  template <class BlockRadixSortT, class KeysT, class ValuesT>
  __device__ void operator()(BlockRadixSortT &block_radix_sort,
                             KeysT &keys,
                             ValuesT &values,
                             int begin_bit,
                             int end_bit,
                             cub::Int2Type<1> /* striped */)
  {
    block_radix_sort.SortDescendingBlockedToStriped(keys,
                                                    values,
                                                    begin_bit,
                                                    end_bit);
  }
};

template <class KeyT>
thrust::host_vector<KeyT>
get_striped_keys(const thrust::host_vector<KeyT> &h_keys,
                 int begin_bit,
                 int end_bit)
{
  thrust::host_vector<KeyT> h_striped_keys(h_keys);
  KeyT *h_striped_keys_data = thrust::raw_pointer_cast(h_striped_keys.data());

  using traits_t                    = cub::Traits<KeyT>;
  using bit_ordered_t               = typename traits_t::UnsignedBits;

  const int num_bits = end_bit - begin_bit;

  for (std::size_t i = 0; i < h_keys.size(); i++)
  {
    bit_ordered_t key = reinterpret_cast<const bit_ordered_t&>(h_keys[i]);

    CUB_IF_CONSTEXPR(traits_t::CATEGORY == cub::FLOATING_POINT)
    {
      const bit_ordered_t negative_zero = bit_ordered_t(1) << bit_ordered_t(sizeof(bit_ordered_t) * 8 - 1);

      if (key == negative_zero)
      {
        key = 0;
      }
    }

    key = traits_t::TwiddleIn(key);

    if ((begin_bit > 0) || (end_bit < static_cast<int>(sizeof(KeyT) * 8)))
    {
      unsigned long long base = 0;
      memcpy(&base, &key, sizeof(bit_ordered_t));
      base &= ((1ULL << num_bits) - 1) << begin_bit;
      memcpy(&key, &base, sizeof(bit_ordered_t));
    }

    // striped keys are used to compare bit ordered representation of keys, 
    // so we do not twiddle-out the key here:
    // key = traits_t::TwiddleOut(key);

    memcpy(h_striped_keys_data + i, &key, sizeof(KeyT));
}

  return h_striped_keys;
}

template <class T>
struct indirect_binary_comparator_t
{
  const T* h_ptr{};
  bool is_descending{};

  indirect_binary_comparator_t(const T* h_ptr, bool is_descending)
      : h_ptr(h_ptr)
      , is_descending(is_descending)
  {}

  bool operator()(std::size_t a, std::size_t b)
  {
    if (is_descending)
    {
      return h_ptr[a] > h_ptr[b];
    }

    return h_ptr[a] < h_ptr[b];
  }
};

template <class KeyT>
thrust::host_vector<std::size_t>
get_permutation(const thrust::host_vector<KeyT> &h_keys,
                bool is_descending,
                int begin_bit,
                int end_bit)
{
  thrust::host_vector<KeyT> h_striped_keys =
    get_striped_keys(h_keys, begin_bit, end_bit);

  thrust::host_vector<std::size_t> h_permutation(h_keys.size());
  thrust::sequence(h_permutation.begin(), h_permutation.end());

  using traits_t = cub::Traits<KeyT>;
  using bit_ordered_t = typename traits_t::UnsignedBits;

  auto bit_ordered_striped_keys =
    reinterpret_cast<const bit_ordered_t*>(thrust::raw_pointer_cast(h_striped_keys.data()));

  std::stable_sort(h_permutation.begin(),
                   h_permutation.end(),
                   indirect_binary_comparator_t<bit_ordered_t>{bit_ordered_striped_keys, is_descending});

  return h_permutation;
}

template <class KeyT>
thrust::host_vector<KeyT>
radix_sort_reference(const thrust::device_vector<KeyT> &d_keys,
                     bool is_descending,
                     int begin_bit,
                     int end_bit)
{
  thrust::host_vector<KeyT> h_keys(d_keys);
  thrust::host_vector<std::size_t> h_permutation =
    get_permutation(h_keys, is_descending, begin_bit, end_bit);
  thrust::host_vector<KeyT> result(d_keys.size());
  thrust::gather(h_permutation.cbegin(), h_permutation.cend(), h_keys.cbegin(), result.begin());

  return result;
}

template <class KeyT, class ValueT>
std::pair<thrust::host_vector<KeyT>, thrust::host_vector<ValueT>>
radix_sort_reference(const thrust::device_vector<KeyT> &d_keys,
                     const thrust::device_vector<ValueT> &d_values,
                     bool is_descending,
                     int begin_bit,
                     int end_bit)
{
  std::pair<thrust::host_vector<KeyT>, thrust::host_vector<ValueT>> result;
  result.first.resize(d_keys.size());
  result.second.resize(d_keys.size());

  thrust::host_vector<KeyT> h_keys(d_keys);
  thrust::host_vector<std::size_t> h_permutation =
    get_permutation(h_keys, is_descending, begin_bit, end_bit);

  thrust::host_vector<ValueT> h_values(d_values);
  thrust::gather(h_permutation.cbegin(),
                 h_permutation.cend(),
                 thrust::make_zip_iterator(h_keys.cbegin(), h_values.cbegin()),
                 thrust::make_zip_iterator(result.first.begin(), result.second.begin()));

  return result;
}
