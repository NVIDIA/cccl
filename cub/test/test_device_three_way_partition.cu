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

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_partition.cuh>
#include <test_util.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>

using namespace cub;

template <typename T>
struct LessThan
{
  T compare;

  explicit __host__ LessThan(T compare)
      : compare(compare)
  {}

  __device__ bool operator()(const T &a) const
  {
    return a < compare;
  }
};

template <typename T>
struct EqualTo
{
  T compare;

  explicit __host__ EqualTo(T compare)
    : compare(compare)
  {}

  __device__ bool operator()(const T &a) const
  {
    return a == compare;
  }
};

template <typename T>
struct GreaterOrEqual
{
  T compare;

  explicit __host__ GreaterOrEqual(T compare)
    : compare(compare)
  {}

  __device__ bool operator()(const T &a) const
  {
    return a >= compare;
  }
};

template <typename T>
void TestEmpty()
{
  int num_items = 0;

  T *in {};
  T *d_first_part_out {};
  T *d_second_part_out {};
  T *d_unselected_out {};
  T *d_num_selected_out {};

  LessThan<T> le(T{0});
  GreaterOrEqual<T> ge(T{1});

  std::size_t temp_storage_size {};
  CubDebugExit(cub::DevicePartition::If(nullptr,
                                        temp_storage_size,
                                        in,
                                        d_first_part_out,
                                        d_second_part_out,
                                        d_unselected_out,
                                        d_num_selected_out,
                                        num_items,
                                        le,
                                        ge));

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);
  std::uint8_t *d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  CubDebugExit(cub::DevicePartition::If(d_temp_storage,
                                        temp_storage_size,
                                        in,
                                        d_first_part_out,
                                        d_second_part_out,
                                        d_unselected_out,
                                        d_num_selected_out,
                                        num_items,
                                        le,
                                        ge));
}

template <typename T>
class ThreeWayPartitionResult
{
public:
  ThreeWayPartitionResult() = delete;
  ThreeWayPartitionResult(int num_items)
    : first_part(num_items)
    , second_part(num_items)
    , unselected(num_items)
  {}

  thrust::device_vector<T> first_part;
  thrust::device_vector<T> second_part;
  thrust::device_vector<T> unselected;

  int num_items_in_first_part {};
  int num_items_in_second_part {};
  int num_unselected_items {};

  bool operator!=(const ThreeWayPartitionResult<T> &other)
  {
    return std::tie(num_items_in_first_part,
                    num_items_in_second_part,
                    num_unselected_items,
                    first_part,
                    second_part,
                    unselected) != std::tie(other.num_items_in_first_part,
                                            other.num_items_in_second_part,
                                            other.num_unselected_items,
                                            other.first_part,
                                            other.second_part,
                                            other.unselected);
  }
};

template <
  typename FirstPartSelectionOp,
  typename SecondPartSelectionOp,
  typename T>
ThreeWayPartitionResult<T> CUBPartition(
  FirstPartSelectionOp first_selector,
  SecondPartSelectionOp second_selector,
  thrust::device_vector<T> &in)
{
  const int num_items = static_cast<int>(in.size());
  ThreeWayPartitionResult<T> result(num_items);

  T *d_in = thrust::raw_pointer_cast(in.data());
  T *d_first_part_out = thrust::raw_pointer_cast(result.first_part.data());
  T *d_second_part_out = thrust::raw_pointer_cast(result.second_part.data());
  T *d_unselected_out = thrust::raw_pointer_cast(result.unselected.data());

  thrust::device_vector<int> num_selected_out(2);
  int *d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  std::size_t temp_storage_size {};
  CubDebugExit(cub::DevicePartition::If(nullptr,
                                        temp_storage_size,
                                        d_in,
                                        d_first_part_out,
                                        d_second_part_out,
                                        d_unselected_out,
                                        d_num_selected_out,
                                        num_items,
                                        first_selector,
                                        second_selector));

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);
  std::uint8_t *d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  CubDebugExit(cub::DevicePartition::If(d_temp_storage,
                                        temp_storage_size,
                                        d_in,
                                        d_first_part_out,
                                        d_second_part_out,
                                        d_unselected_out,
                                        d_num_selected_out,
                                        num_items,
                                        first_selector,
                                        second_selector));

  thrust::host_vector<int> h_num_selected_out(num_selected_out);

  result.num_items_in_first_part = h_num_selected_out[0];
  result.num_items_in_second_part = h_num_selected_out[1];

  result.num_unselected_items = num_items
                              - h_num_selected_out[0]
                              - h_num_selected_out[1];

  return result;
}

template <
  typename FirstPartSelectionOp,
  typename SecondPartSelectionOp,
  typename T>
ThreeWayPartitionResult<T> ThrustPartition(
  FirstPartSelectionOp first_selector,
  SecondPartSelectionOp second_selector,
  thrust::device_vector<T> &in)
{
  const int num_items = static_cast<int>(in.size());
  ThreeWayPartitionResult<T> result(num_items);

  thrust::device_vector<T> intermediate_result(num_items);

  auto intermediate_iterators =
    thrust::partition_copy(in.begin(),
                           in.end(),
                           result.first_part.begin(),
                           intermediate_result.begin(),
                           first_selector);

  result.num_items_in_first_part = static_cast<int>(
    thrust::distance(result.first_part.begin(), intermediate_iterators.first));

  auto final_iterators = thrust::partition_copy(
    intermediate_result.begin(),
    intermediate_result.begin() + (num_items - result.num_items_in_first_part),
    result.second_part.begin(),
    result.unselected.begin(),
    second_selector);

  result.num_items_in_second_part = static_cast<int>(
    thrust::distance(result.second_part.begin(), final_iterators.first));

  result.num_unselected_items = static_cast<int>(
    thrust::distance(result.unselected.begin(), final_iterators.second));

  return result;
}

template <typename T>
void TestEmptyFirstPart(int num_items)
{
  thrust::device_vector<T> in(num_items);
  thrust::sequence(in.begin(), in.end());

  T first_unselected_val = T{0};
  T first_val_of_second_part = static_cast<T>(num_items / 2);

  LessThan<T> le(first_unselected_val);
  GreaterOrEqual<T> ge(first_val_of_second_part);

  auto cub_result = CUBPartition(le, ge, in);
  auto thrust_result = ThrustPartition(le, ge, in);

  AssertEquals(cub_result, thrust_result);
  AssertEquals(cub_result.num_items_in_first_part, 0);
}

template <typename T>
void TestEmptySecondPart(int num_items)
{
  thrust::device_vector<T> in(num_items);
  thrust::sequence(in.begin(), in.end());

  T first_unselected_val = static_cast<T>(num_items / 2);
  T first_val_of_second_part = T{0}; // empty set for unsigned types

  GreaterOrEqual<T> ge(first_unselected_val);
  LessThan<T> le(first_val_of_second_part);

  auto cub_result = CUBPartition(ge, le, in);
  auto thrust_result = ThrustPartition(ge, le, in);

  AssertEquals(cub_result, thrust_result);
  AssertEquals(cub_result.num_items_in_second_part, 0);
}

template <typename T>
void TestEmptyUnselectedPart(int num_items)
{
  thrust::device_vector<T> in(num_items);
  thrust::sequence(in.begin(), in.end());

  T first_unselected_val = static_cast<T>(num_items / 2);

  LessThan<T> le(first_unselected_val);
  GreaterOrEqual<T> ge(first_unselected_val);

  auto cub_result = CUBPartition(le, ge, in);
  auto thrust_result = ThrustPartition(le, ge, in);

  AssertEquals(cub_result, thrust_result);
  AssertEquals(cub_result.num_unselected_items, 0);
}

template <typename T>
void TestUnselectedOnly(int num_items)
{
  thrust::device_vector<T> in(num_items);
  thrust::sequence(in.begin(), in.end());

  T first_val_of_second_part = T{0}; // empty set for unsigned types
  LessThan<T> le(first_val_of_second_part);

  auto cub_result = CUBPartition(le, le, in);
  auto thrust_result = ThrustPartition(le, le, in);

  AssertEquals(cub_result, thrust_result);
  AssertEquals(cub_result.num_unselected_items, num_items);
  AssertEquals(cub_result.num_items_in_first_part, 0);
  AssertEquals(cub_result.num_items_in_second_part, 0);
}

template <typename Key,
          typename Value>
struct Pair
{
  Key key;
  Value value;

  __host__ __device__ Pair()
      : key(Key{})
      , value(Value{})
  {}

  __host__ __device__ Pair(Key key)
    : key(key)
    , value(Value{})
  {}

  __host__ __device__ Pair(Key key, Value value)
    : key(key)
    , value(value)
  {}

  __host__ __device__ bool operator<(const Pair &b) const
  {
    return key < b.key;
  }

  __host__ __device__ bool operator>=(const Pair &b) const
  {
    return key >= b.key;
  }
};

template <typename Key, typename Value>
__device__ __host__ bool operator==(
  const Pair<Key, Value> &lhs,
  const Pair<Key, Value> &rhs)
{
  return lhs.key == rhs.key && lhs.value == lhs.value;
}

template <typename ValueT>
struct CountToPair
{
  template <typename OffsetT>
  __device__ __host__ Pair<ValueT, std::uint32_t> operator()(OffsetT id)
  {
    return Pair<ValueT, std::uint32_t>(static_cast<ValueT>(id), id);
  }
};

template <typename KeyT>
void TestStability(int num_items)
{
  using T = Pair<KeyT, std::uint32_t>;
  thrust::device_vector<T> in(num_items);

  thrust::tabulate(in.begin(), in.end(), CountToPair<KeyT>{});

  T first_unselected_val = static_cast<KeyT>(num_items / 3);
  T first_val_of_second_part = static_cast<KeyT>(2 * num_items / 3);

  LessThan<T> le(first_unselected_val);
  GreaterOrEqual<T> ge(first_val_of_second_part);

  auto cub_result = CUBPartition(le, ge, in);
  auto thrust_result = ThrustPartition(le, ge, in);

  AssertEquals(cub_result, thrust_result);
}

template <typename T>
void TestReverseIterator(int num_items)
{
  int num_items_in_first_part = num_items / 3;
  int num_unselected_items = 2 * num_items / 3;

  T first_part_val {0};
  T second_part_val {1};
  T unselected_part_val {2};

  thrust::device_vector<T> in(num_items, second_part_val);
  thrust::fill_n(in.begin(), num_items_in_first_part, first_part_val);
  thrust::fill_n(in.begin() + num_items_in_first_part,
                 num_unselected_items,
                 unselected_part_val);

  thrust::shuffle(in.begin(), in.end(), thrust::default_random_engine{});

  thrust::device_vector<T> first_and_unselected_part(num_items);

  EqualTo<T> first_selector{first_part_val};
  EqualTo<T> second_selector{second_part_val};

  thrust::device_vector<int> num_selected_out(2);

  std::size_t temp_storage_size {};
  CubDebugExit(cub::DevicePartition::If(nullptr,
                                        temp_storage_size,
                                        in.cbegin(),
                                        first_and_unselected_part.begin(),
                                        thrust::make_discard_iterator(),
                                        first_and_unselected_part.rbegin(),
                                        num_selected_out.begin(),
                                        num_items,
                                        first_selector,
                                        second_selector));

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);
  std::uint8_t *d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  CubDebugExit(cub::DevicePartition::If(d_temp_storage,
                                        temp_storage_size,
                                        in.cbegin(),
                                        first_and_unselected_part.begin(),
                                        thrust::make_discard_iterator(),
                                        first_and_unselected_part.rbegin(),
                                        num_selected_out.begin(),
                                        num_items,
                                        first_selector,
                                        second_selector));

  thrust::device_vector<int> h_num_selected_out(num_selected_out);

  AssertEquals(h_num_selected_out[0], num_items_in_first_part);

  AssertEquals(thrust::count(first_and_unselected_part.rbegin(),
                             first_and_unselected_part.rbegin() +
                               num_unselected_items,
                             unselected_part_val),
               num_unselected_items);

  AssertEquals(thrust::count(first_and_unselected_part.begin(),
                             first_and_unselected_part.begin() +
                               num_items_in_first_part,
                             first_part_val),
               num_items_in_first_part);
}

template <typename T>
void TestSingleOutput(int num_items)
{
  int num_items_in_first_part = num_items / 3;
  int num_unselected_items = 2 * num_items / 3;
  int num_items_in_second_part = num_items - num_items_in_first_part -
                                 num_unselected_items;

  T first_part_val{0};
  T second_part_val{1};
  T unselected_part_val{2};

  thrust::device_vector<T> in(num_items, second_part_val);
  thrust::fill_n(in.begin(), num_items_in_first_part, first_part_val);
  thrust::fill_n(in.begin() + num_items_in_first_part,
                 num_unselected_items,
                 unselected_part_val);

  thrust::shuffle(in.begin(), in.end(), thrust::default_random_engine{});

  thrust::device_vector<T> output(num_items);

  EqualTo<T> first_selector{first_part_val};
  EqualTo<T> second_selector{second_part_val};

  thrust::device_vector<int> num_selected_out(2);

  std::size_t temp_storage_size{};
  CubDebugExit(cub::DevicePartition::If(nullptr,
                                        temp_storage_size,
                                        in.cbegin(),
                                        output.begin(),
                                        output.begin() + num_items_in_first_part,
                                        output.rbegin(),
                                        num_selected_out.begin(),
                                        num_items,
                                        first_selector,
                                        second_selector));

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);
  std::uint8_t *d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  CubDebugExit(cub::DevicePartition::If(d_temp_storage,
                                        temp_storage_size,
                                        in.cbegin(),
                                        output.begin(),
                                        output.begin() + num_items_in_first_part,
                                        output.rbegin(),
                                        num_selected_out.begin(),
                                        num_items,
                                        first_selector,
                                        second_selector));

  thrust::device_vector<int> h_num_selected_out(num_selected_out);

  AssertEquals(h_num_selected_out[0], num_items_in_first_part);
  AssertEquals(h_num_selected_out[1], num_items_in_second_part);

  AssertEquals(thrust::count(output.rbegin(),
                             output.rbegin() + num_unselected_items,
                             unselected_part_val),
               num_unselected_items);

  AssertEquals(thrust::count(output.begin(),
                             output.begin() + num_items_in_first_part,
                             first_part_val),
               num_items_in_first_part);

  AssertEquals(thrust::count(output.begin() + num_items_in_first_part,
                             output.begin() + num_items_in_first_part +
                               num_items_in_second_part,
                             second_part_val),
               num_items_in_second_part);
}

template <typename T>
void TestNumItemsDependent(int num_items)
{
  TestStability<T>(num_items);
  TestEmptyFirstPart<T>(num_items);
  TestEmptySecondPart<T>(num_items);
  TestEmptyUnselectedPart<T>(num_items);
  TestUnselectedOnly<T>(num_items);
  TestReverseIterator<T>(num_items);
  TestSingleOutput<T>(num_items);
}

template <typename T>
void TestNumItemsDependent()
{
  for (int num_items = 1; num_items < 1000000; num_items <<= 2)
  {
    TestNumItemsDependent<T>(num_items);
    TestNumItemsDependent<T>(num_items + 31);
  }
}

template <typename T>
void Test()
{
  TestEmpty<T>();
  TestNumItemsDependent<T>();
}

int main(int argc, char **argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // NVBug 4136386
  // Test<std::uint8_t>();
  // Test<std::uint16_t>();
  Test<std::uint32_t>();
  Test<std::uint64_t>();

  return 0;
}
