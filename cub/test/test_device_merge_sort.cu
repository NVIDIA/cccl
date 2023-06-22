/******************************************************************************
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

/******************************************************************************
 * Test of DeviceMergeSort utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_merge_sort.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "test_util.h"

#include <cstdio>
#include <limits>
#include <new> // for std::bad_alloc
#include <memory>
#include <typeinfo>

using namespace cub;

struct CustomLess
{
  template <typename DataType>
  __device__ bool operator()(DataType &lhs, DataType &rhs)
  {
    return lhs < rhs;
  }
};

template <typename DataType>
bool CheckResult(thrust::device_vector<DataType> &d_data)
{
  const bool is_sorted = thrust::is_sorted(d_data.begin(), d_data.end(), CustomLess());
  return is_sorted;
}

template <typename KeyType, typename ValueType>
struct ValueToKey
{
  __device__ __host__ KeyType operator()(const ValueType &val)
  {
    return val;
  }
};

template <typename ValueType>
struct ValueToKey<HugeDataType, ValueType>
{
  __device__ __host__ HugeDataType operator()(const ValueType &val)
  {
    return HugeDataType(val);
  }
};

template <typename KeyType, typename DataType>
void Test(std::int64_t num_items,
          thrust::default_random_engine &rng,
          thrust::device_vector<KeyType> &d_keys,
          thrust::device_vector<DataType> &d_values)
{
  thrust::sequence(d_values.begin(), d_values.end());
  thrust::shuffle(d_values.begin(), d_values.end(), rng);

  thrust::transform(d_values.begin(),
                    d_values.end(),
                    d_keys.begin(),
                    ValueToKey<KeyType, DataType>());

  thrust::device_vector<KeyType> d_keys_before_sort(d_keys);
  thrust::device_vector<DataType> d_values_before_sort(d_values);

  thrust::device_vector<KeyType> d_keys_before_sort_copy(d_keys);
  thrust::device_vector<DataType> d_values_before_sort_copy(d_values);

  size_t temp_size = 0;
  CubDebugExit(cub::DeviceMergeSort::SortPairs(
      nullptr,
      temp_size,
      thrust::raw_pointer_cast(d_keys.data()),
      thrust::raw_pointer_cast(d_values.data()),
      num_items,
      CustomLess()));

  thrust::device_vector<char> tmp(temp_size);

  CubDebugExit(cub::DeviceMergeSort::SortPairs(
    thrust::raw_pointer_cast(tmp.data()),
    temp_size,
    thrust::raw_pointer_cast(d_keys.data()),
    thrust::raw_pointer_cast(d_values.data()),
    num_items,
    CustomLess()));

  thrust::device_vector<KeyType> d_keys_after_sort_copy(d_keys);
  thrust::device_vector<DataType> d_values_after_sort_copy(d_values);

  AssertTrue(CheckResult(d_values));

  CubDebugExit(cub::DeviceMergeSort::SortPairsCopy(
    thrust::raw_pointer_cast(tmp.data()),
    temp_size,
    thrust::raw_pointer_cast(d_keys_before_sort.data()),
    thrust::raw_pointer_cast(d_values_before_sort.data()),
    thrust::raw_pointer_cast(d_keys.data()),
    thrust::raw_pointer_cast(d_values.data()),
    num_items,
    CustomLess()));

  AssertEquals(d_keys, d_keys_after_sort_copy);
  AssertEquals(d_values, d_values_after_sort_copy);
  AssertEquals(d_keys_before_sort, d_keys_before_sort_copy);
  AssertEquals(d_values_before_sort, d_values_before_sort_copy);

  // At the moment stable sort is an alias to sort, so it's safe to use
  // temp_size storage allocated before
  CubDebugExit(cub::DeviceMergeSort::StableSortPairs(
    thrust::raw_pointer_cast(tmp.data()),
    temp_size,
    thrust::raw_pointer_cast(d_keys.data()),
    thrust::raw_pointer_cast(d_values.data()),
    num_items,
    CustomLess()));

  AssertTrue(CheckResult(d_values));

  CubDebugExit(cub::DeviceMergeSort::SortPairsCopy(
    thrust::raw_pointer_cast(tmp.data()),
    temp_size,
    thrust::constant_iterator<KeyType>(KeyType(42)),
    thrust::counting_iterator<DataType>(DataType(0)),
    thrust::raw_pointer_cast(d_keys.data()),
    thrust::raw_pointer_cast(d_values.data()),
    num_items,
    CustomLess()));

  thrust::sequence(d_values_before_sort.begin(), d_values_before_sort.end());

  AssertEquals(d_values, d_values_before_sort);
}

template <typename KeyType, typename DataType>
void TestKeys(std::int64_t num_items,
              thrust::default_random_engine &rng,
              thrust::device_vector<KeyType> &d_keys,
              thrust::device_vector<DataType> &d_values)
{
  thrust::sequence(d_values.begin(), d_values.end());
  thrust::shuffle(d_values.begin(), d_values.end(), rng);

  thrust::transform(d_values.begin(),
                    d_values.end(),
                    d_keys.begin(),
                    ValueToKey<KeyType, DataType>());

  thrust::device_vector<KeyType> d_before_sort(d_keys);
  thrust::device_vector<KeyType> d_before_sort_copy(d_keys);

  size_t temp_size = 0;
  cub::DeviceMergeSort::SortKeys(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(d_keys.data()),
    num_items,
    CustomLess());

  thrust::device_vector<char> tmp(temp_size);

  CubDebugExit(cub::DeviceMergeSort::SortKeys(
    thrust::raw_pointer_cast(tmp.data()),
    temp_size,
    thrust::raw_pointer_cast(d_keys.data()),
    num_items,
    CustomLess()));

  thrust::device_vector<KeyType> d_after_sort(d_keys);

  AssertTrue(CheckResult(d_keys));

  CubDebugExit(cub::DeviceMergeSort::SortKeysCopy(
    thrust::raw_pointer_cast(tmp.data()),
    temp_size,
    thrust::raw_pointer_cast(d_before_sort.data()),
    thrust::raw_pointer_cast(d_keys.data()),
    num_items,
    CustomLess()));

  AssertTrue(d_keys == d_after_sort);
  AssertTrue(d_before_sort == d_before_sort_copy);

  // At the moment stable sort is an alias to sort, so it's safe to use
  // temp_size storage allocated before
  CubDebugExit(cub::DeviceMergeSort::StableSortKeys(
    thrust::raw_pointer_cast(tmp.data()),
    temp_size,
    thrust::raw_pointer_cast(d_keys.data()),
    num_items,
    CustomLess()));

  AssertTrue(CheckResult(d_keys));

  thrust::fill(d_keys.begin(), d_keys.end(), KeyType{});
  CubDebugExit(cub::DeviceMergeSort::StableSortKeysCopy(
      thrust::raw_pointer_cast(tmp.data()),
      temp_size,
      thrust::raw_pointer_cast(d_before_sort.data()),
      thrust::raw_pointer_cast(d_keys.data()),
      num_items,
      CustomLess()));

  // AssertTrue(CheckResult(d_keys));
  AssertTrue(d_keys == d_after_sort);
  AssertTrue(d_before_sort == d_before_sort_copy);
}

template <bool data_dont_exceed_key_size>
struct TestHelper
{
  template <typename KeyType, typename DataType>
  static void AllocateAndTest(thrust::default_random_engine &rng, unsigned int num_items)
  {
    thrust::device_vector<KeyType> d_keys(num_items);
    thrust::device_vector<DataType> d_values(num_items);

    Test<KeyType, DataType>(num_items, rng, d_keys, d_values);
    TestKeys<KeyType, DataType>(num_items, rng, d_keys, d_values);
  }
};

template <>
struct TestHelper<false>
{
  template <typename, typename>
  static void AllocateAndTest(thrust::default_random_engine &, unsigned int)
  {}
};

template <typename DataType>
void Test(thrust::default_random_engine &rng, unsigned int num_items)
{
  TestHelper<sizeof(DataType) <= sizeof(std::uint8_t) >::template AllocateAndTest<std::uint8_t,  DataType>(rng, num_items);
  TestHelper<sizeof(DataType) <= sizeof(std::uint32_t)>::template AllocateAndTest<std::uint32_t, DataType>(rng, num_items);
  TestHelper<sizeof(DataType) <= sizeof(std::uint64_t)>::template AllocateAndTest<std::uint64_t, DataType>(rng, num_items);
}

template <typename KeyType, typename DataType>
void AllocateAndTestIterators(unsigned int num_items)
{
  thrust::device_vector<KeyType> d_keys(num_items);
  thrust::device_vector<DataType> d_values(num_items);

  thrust::sequence(d_keys.begin(), d_keys.end());
  thrust::sequence(d_values.begin(), d_values.end());

  thrust::reverse(d_values.begin(), d_values.end());

  using KeyIterator = typename thrust::device_vector<KeyType>::iterator;
  thrust::reverse_iterator<KeyIterator> reverse_iter(d_keys.end());

  size_t temp_size = 0;
  cub::DeviceMergeSort::SortPairs(
    nullptr,
    temp_size,
    reverse_iter,
    thrust::raw_pointer_cast(d_values.data()),
    num_items,
    CustomLess());

  thrust::device_vector<char> tmp(temp_size);

  cub::DeviceMergeSort::SortPairs(
    thrust::raw_pointer_cast(tmp.data()),
    temp_size,
    reverse_iter,
    thrust::raw_pointer_cast(d_values.data()),
    num_items,
    CustomLess());

  AssertTrue(CheckResult(d_values));
}

template <typename DataType>
void Test(thrust::default_random_engine &rng)
{
  for (unsigned int pow2 = 9; pow2 < 22; pow2 += 2)
  {
    try
    {
      const unsigned int num_items = 1 << pow2;
      AllocateAndTestIterators<DataType, DataType>(num_items);

      TestHelper<true>::AllocateAndTest<HugeDataType, DataType>(rng, num_items);
      Test<DataType>(rng, num_items);
    }
    catch (std::bad_alloc &e)
    {
      if (pow2 > 20)
      { // Some cards don't have enough memory for large allocations, these
        // can be skipped.
        printf("Skipping large memory test. (num_items=2^%u): %s\n",
               pow2,
               e.what());
      }
      else
      { // For smaller problem sizes, treat as an error:
        printf("Error (num_items=2^%u): %s", pow2, e.what());
        throw;
      }
    }
  }
}

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  thrust::default_random_engine rng;

  Test<std::int32_t>(rng);
  Test<std::int64_t>(rng);

  return 0;
}
