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

#include "test_util.h"
#include "cub/thread/thread_sort.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/random.h>


struct CustomLess
{
  template <typename DataType>
  __host__ __device__ bool operator()(DataType &lhs, DataType &rhs)
  {
    return lhs < rhs;
  }
};


template <typename KeyT,
          typename ValueT,
          int ItemsPerThread>
__global__ void kernel(const KeyT *keys_in,
                       KeyT *keys_out,
                       const ValueT *values_in,
                       ValueT *values_out)
{
  KeyT thread_keys[ItemsPerThread];
  KeyT thread_values[ItemsPerThread];

  const auto thread_offset = ItemsPerThread * threadIdx.x;
  keys_in += thread_offset;
  keys_out += thread_offset;
  values_in += thread_offset;
  values_out += thread_offset;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    thread_keys[item] = keys_in[item];
    thread_values[item] = values_in[item];
  }

  cub::StableOddEvenSort(thread_keys, thread_values, CustomLess{});

  for (int item = 0; item < ItemsPerThread; item++)
  {
    keys_out[item] = thread_keys[item];
    values_out[item] = thread_values[item];
  }
}


template <typename KeyT,
          typename ValueT,
          int ItemsPerThread>
void Test()
{
  constexpr unsigned int threads_in_block = 1024;
  constexpr unsigned int elements = threads_in_block * ItemsPerThread;

  thrust::default_random_engine re;
  thrust::device_vector<std::uint8_t> data_source(elements);

  for (int iteration = 0; iteration < 10; iteration++)
  {
    thrust::sequence(data_source.begin(), data_source.end());
    thrust::shuffle(data_source.begin(), data_source.end(), re);
    thrust::device_vector<KeyT> in_keys(data_source);
    thrust::device_vector<KeyT> out_keys(elements);

    thrust::shuffle(data_source.begin(), data_source.end(), re);
    thrust::device_vector<ValueT> in_values(data_source);
    thrust::device_vector<ValueT> out_values(elements);

    thrust::host_vector<KeyT> host_keys(in_keys);
    thrust::host_vector<ValueT> host_values(in_values);

    kernel<KeyT, ValueT, ItemsPerThread><<<1, threads_in_block>>>(
      thrust::raw_pointer_cast(in_keys.data()),
      thrust::raw_pointer_cast(out_keys.data()),
      thrust::raw_pointer_cast(in_values.data()),
      thrust::raw_pointer_cast(out_values.data()));

    for (unsigned int tid = 0; tid < threads_in_block; tid++)
    {
      const auto thread_begin = tid * ItemsPerThread;
      const auto thread_end = thread_begin + ItemsPerThread;

      thrust::sort_by_key(host_keys.begin() + thread_begin,
                          host_keys.begin() + thread_end,
                          host_values.begin() + thread_begin,
                          CustomLess{});
    }

    AssertEquals(host_keys, out_keys);
    AssertEquals(host_values, out_values);
  }
}


template <typename KeyT,
          typename ValueT>
void Test()
{
  Test<KeyT, ValueT, 2>();
  Test<KeyT, ValueT, 3>();
  Test<KeyT, ValueT, 4>();
  Test<KeyT, ValueT, 5>();
  Test<KeyT, ValueT, 7>();
  Test<KeyT, ValueT, 8>();
  Test<KeyT, ValueT, 9>();
  Test<KeyT, ValueT, 11>();
}

int main()
{
  Test<std::uint32_t, std::uint32_t>();
  Test<std::uint32_t, std::uint64_t>();

  return 0;
}
