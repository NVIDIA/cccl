//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "build_result_caching.h"
#include "test_util.h"
#include <cccl/c/three_way_partition.h>

using BuildResultT = cccl_device_three_way_partition_build_result_t;

struct three_way_partition_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_three_way_partition_cleanup(build_data);
  }
};

using three_way_partition_deleter = BuildResultDeleter<BuildResultT, three_way_partition_cleanup>;
using three_way_partition_build_cache_t =
  build_cache_t<std::string, result_wrapper_t<BuildResultT, three_way_partition_deleter>>;

template <typename KeyType, typename NumSelectedType>
struct TestParameters
{
  using KeyT         = KeyType;
  using NumSelectedT = NumSelectedType;
};

template <typename Tag>
auto& get_cache()
{
  return fixture<three_way_partition_build_cache_t, Tag>::get_or_create().get_value();
}

struct three_way_partition_build
{
  template <typename... Rest>
  CUresult operator()(
    BuildResultT* build_ptr,
    cccl_iterator_t d_in,
    cccl_iterator_t d_first_part_out,
    cccl_iterator_t d_second_part_out,
    cccl_iterator_t d_unselected_out,
    cccl_iterator_t d_num_selected_out,
    cccl_op_t select_first_part_op,
    cccl_op_t select_second_part_op,
    int64_t /*num_items*/,
    Rest... rest) const noexcept
  {
    return cccl_device_three_way_partition_build(
      build_ptr,
      d_in,
      d_first_part_out,
      d_second_part_out,
      d_unselected_out,
      d_num_selected_out,
      select_first_part_op,
      select_second_part_op,
      rest...);
  }
};

struct three_way_partition_run
{
  template <typename... Args>
  CUresult operator()(Args... args) const noexcept
  {
    return cccl_device_three_way_partition(args...);
  }
};

// Host-side reference implementation using the C++ standard library
template <typename T>
struct three_way_partition_result_t
{
  three_way_partition_result_t() = delete;
  explicit three_way_partition_result_t(int num_items)
      : first_part(num_items)
      , second_part(num_items)
      , unselected(num_items)
  {}
  explicit three_way_partition_result_t(
    std::vector<T> first, std::vector<T> second, std::vector<T> unselected, int n_first, int n_second, int n_unselected)
      : first_part(std::move(first))
      , second_part(std::move(second))
      , unselected(std::move(unselected))
      , num_items_in_first_part(n_first)
      , num_items_in_second_part(n_second)
      , num_unselected_items(n_unselected)
  {}

  std::vector<T> first_part;
  std::vector<T> second_part;
  std::vector<T> unselected;

  int num_items_in_first_part{};
  int num_items_in_second_part{};
  int num_unselected_items{};

  bool operator==(const three_way_partition_result_t<T>& other) const
  {
    return std::tie(num_items_in_first_part,
                    num_items_in_second_part,
                    num_unselected_items,
                    first_part,
                    second_part,
                    unselected)
        == std::tie(other.num_items_in_first_part,
                    other.num_items_in_second_part,
                    other.num_unselected_items,
                    other.first_part,
                    other.second_part,
                    other.unselected);
  }
};

template <typename T>
struct greater_or_equal_t
{
  T compare;

  explicit __host__ greater_or_equal_t(T compare)
      : compare(compare)
  {}

  __device__ bool operator()(const T& a) const
  {
    return a >= compare;
  }
};

template <typename T>
struct less_than_t
{
  T compare;

  explicit __host__ less_than_t(T compare)
      : compare(compare)
  {}

  __device__ bool operator()(const T& a) const
  {
    return a < compare;
  }
};

template <typename FirstPartSelectionOp, typename SecondPartSelectionOp, typename T>
three_way_partition_result_t<T>
std_partition(FirstPartSelectionOp first_selector, SecondPartSelectionOp second_selector, const std::vector<T>& in)
{
  const int num_items = static_cast<int>(in.size());
  three_way_partition_result_t<T> result(num_items);

  std::vector<T> intermediate_result(num_items);

  auto intermediate_iterators =
    std::partition_copy(in.begin(), in.end(), result.first_part.begin(), intermediate_result.begin(), first_selector);

  result.num_items_in_first_part =
    static_cast<int>(std::distance(result.first_part.begin(), intermediate_iterators.first));

  auto final_iterators = std::partition_copy(
    intermediate_result.begin(),
    intermediate_result.begin() + (num_items - result.num_items_in_first_part),
    result.second_part.begin(),
    result.unselected.begin(),
    second_selector);

  result.num_items_in_second_part = static_cast<int>(std::distance(result.second_part.begin(), final_iterators.first));
  result.num_unselected_items     = static_cast<int>(std::distance(result.unselected.begin(), final_iterators.second));

  return result;
}

template <typename KeyT, typename NumSelectedT, typename TagT>
three_way_partition_result_t<KeyT>
c_parallel_partition(operation_t first_selector, operation_t second_selector, const std::vector<KeyT>& input)
{
  std::size_t num_items = input.size();

  pointer_t<KeyT> input_ptr(input);
  pointer_t<KeyT> first_part_output_ptr(num_items);
  pointer_t<KeyT> second_part_output_ptr(num_items);
  pointer_t<KeyT> unselected_output_ptr(num_items);
  pointer_t<NumSelectedT> num_selected_ptr(2);

  auto& build_cache    = get_cache<TagT>();
  const auto& test_key = make_key<KeyT, NumSelectedT>();

  three_way_partition(
    input_ptr,
    first_part_output_ptr,
    second_part_output_ptr,
    unselected_output_ptr,
    num_selected_ptr,
    first_selector,
    second_selector,
    num_items,
    build_cache,
    test_key);

  std::vector<KeyT> first_part_output(first_part_output_ptr);
  std::vector<KeyT> second_part_output(second_part_output_ptr);
  std::vector<KeyT> unselected_output(unselected_output_ptr);
  std::vector<NumSelectedT> num_selected(num_selected_ptr);

  return three_way_partition_result_t<KeyT>(
    std::move(first_part_output),
    std::move(second_part_output),
    std::move(unselected_output),
    num_selected[0],
    num_selected[1],
    num_items - num_selected[0] - num_selected[1]);
}

template <typename BuildCache = three_way_partition_build_cache_t, typename KeyT = std::string>
void three_way_partition(
  cccl_iterator_t d_in,
  cccl_iterator_t d_first_part_out,
  cccl_iterator_t d_second_part_out,
  cccl_iterator_t d_unselected_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t select_first_part_op,
  cccl_op_t select_second_part_op,
  int64_t num_items,
  std::optional<BuildCache>& cache,
  const std::optional<KeyT>& lookup_key)
{
  AlgorithmExecute<BuildResultT,
                   three_way_partition_build,
                   three_way_partition_cleanup,
                   three_way_partition_run,
                   BuildCache,
                   KeyT>(
    cache,
    lookup_key,
    d_in,
    d_first_part_out,
    d_second_part_out,
    d_unselected_out,
    d_num_selected_out,
    select_first_part_op,
    select_second_part_op,
    num_items);
}

// ==============
//   Test section
// ==============

using key_types =
  c2h::type_list<uint8_t,
                 int16_t,
                 uint32_t,
                 int64_t,
                 uint64_t,
#if _CCCL_HAS_NVFP16()
                 __half,
#endif
                 float,
                 double>;

using num_selected_types = c2h::type_list<uint32_t, int64_t>;

using test_params_tuple =
  c2h::type_list<TestParameters<c2h::get<0, key_types>, c2h::get<0, num_selected_types>>,
                 TestParameters<c2h::get<1, key_types>, c2h::get<1, num_selected_types>>,
                 TestParameters<c2h::get<2, key_types>, c2h::get<0, num_selected_types>>,
                 TestParameters<c2h::get<3, key_types>, c2h::get<1, num_selected_types>>,
                 TestParameters<c2h::get<4, key_types>, c2h::get<0, num_selected_types>>,
                 TestParameters<c2h::get<5, key_types>, c2h::get<1, num_selected_types>>>;

struct ThreeWayPartition_PrimitiveTypes_Fixture_Tag;
C2H_TEST("ThreeWayPartition works with primitive types", "[three_way_partition]", test_params_tuple)
{
  using T              = c2h::get<0, TestType>;
  using key_t          = T::KeyT;
  using num_selected_t = T::NumSelectedT;

  auto [less_op_src, greater_or_equal_op_src] = get_three_way_partition_ops(get_type_info<key_t>().type, 42);
  operation_t less_op                         = make_operation("less_op", less_op_src);
  operation_t greater_or_equal_op             = make_operation("greater_op", greater_or_equal_op_src);

  const std::size_t num_items      = GENERATE(0, 42, take(4, random(1 << 12, 1 << 20)));
  const std::vector<int> input_int = generate<int>(num_items);
  const std::vector<key_t> input(input_int.begin(), input_int.end());

  auto c_parallel_result = c_parallel_partition<key_t, num_selected_t, ThreeWayPartition_PrimitiveTypes_Fixture_Tag>(
    less_op, greater_or_equal_op, input);
  auto std_result = std_partition(less_than_t<key_t>{key_t{42}}, greater_or_equal_t<key_t>{key_t{42}}, input);

  REQUIRE(c_parallel_result == std_result);
}

// stateful operations test
// custom types test
// iterator test
// well-known operations
