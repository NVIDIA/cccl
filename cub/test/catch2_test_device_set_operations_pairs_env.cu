// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_set_operations.cuh>

#include <algorithm>

#include "block_size_extracting_helpers.h"
#include "catch2_test_launch_helper.h"

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSetOps::SetDifferencePairs, set_difference_pairs);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSetOps::SetIntersectionPairs, set_intersection_pairs);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSetOps::SetSymmetricDifferencePairs, set_symmetric_difference_pairs);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSetOps::SetUnionPairs, set_union_pairs);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

// One tag per by-key set operation, bundling the launch wrapper (single-phase env API), the explicit-temp-storage API
// (used to query the temporary-storage size), and the std reference algorithm operating on keys only.
struct op_difference
{
  static constexpr bool values_from_first_input_only = true;
  template <typename... Args>
  static void wrapper(Args&&... args)
  {
    set_difference_pairs(::cuda::std::forward<Args>(args)...);
  }
  template <typename... Args>
  static cudaError_t api(Args&&... args)
  {
    return cub::DeviceSetOps::SetDifferencePairs(::cuda::std::forward<Args>(args)...);
  }
  template <typename It, typename Out>
  static void reference(It a1, It a2, It b1, It b2, Out out)
  {
    std::set_difference(a1, a2, b1, b2, out);
  }
};
struct op_intersection
{
  static constexpr bool values_from_first_input_only = true;
  template <typename... Args>
  static void wrapper(Args&&... args)
  {
    set_intersection_pairs(::cuda::std::forward<Args>(args)...);
  }
  template <typename... Args>
  static cudaError_t api(Args&&... args)
  {
    return cub::DeviceSetOps::SetIntersectionPairs(::cuda::std::forward<Args>(args)...);
  }
  template <typename It, typename Out>
  static void reference(It a1, It a2, It b1, It b2, Out out)
  {
    std::set_intersection(a1, a2, b1, b2, out);
  }
};
struct op_symmetric_difference
{
  static constexpr bool values_from_first_input_only = false;
  template <typename... Args>
  static void wrapper(Args&&... args)
  {
    set_symmetric_difference_pairs(::cuda::std::forward<Args>(args)...);
  }
  template <typename... Args>
  static cudaError_t api(Args&&... args)
  {
    return cub::DeviceSetOps::SetSymmetricDifferencePairs(::cuda::std::forward<Args>(args)...);
  }
  template <typename It, typename Out>
  static void reference(It a1, It a2, It b1, It b2, Out out)
  {
    std::set_symmetric_difference(a1, a2, b1, b2, out);
  }
};
struct op_union
{
  static constexpr bool values_from_first_input_only = false;
  template <typename... Args>
  static void wrapper(Args&&... args)
  {
    set_union_pairs(::cuda::std::forward<Args>(args)...);
  }
  template <typename... Args>
  static cudaError_t api(Args&&... args)
  {
    return cub::DeviceSetOps::SetUnionPairs(::cuda::std::forward<Args>(args)...);
  }
  template <typename It, typename Out>
  static void reference(It a1, It a2, It b1, It b2, Out out)
  {
    std::set_union(a1, a2, b1, b2, out);
  }
};

using key_ops = c2h::type_list<op_difference, op_intersection, op_symmetric_difference, op_union>;

// Values are tagged as key*2 (first input) or key*2+1 (second input) so provenance can be verified.
inline auto values_for(const c2h::device_vector<int>& keys, int source_bit) -> c2h::device_vector<int>
{
  c2h::host_vector<int> keys_h = keys;
  c2h::host_vector<int> values(keys_h.size());
  for (std::size_t i = 0; i < keys_h.size(); ++i)
  {
    values[i] = keys_h[i] * 2 + source_bit;
  }
  return c2h::device_vector<int>(values);
}

template <typename Op>
void check_pairs(const c2h::device_vector<int>& keys1,
                 const c2h::device_vector<int>& keys2,
                 const c2h::device_vector<int>& keys_out,
                 const c2h::device_vector<int>& values_out,
                 const c2h::device_vector<int>& num_selected)
{
  c2h::host_vector<int> h1 = keys1;
  c2h::host_vector<int> h2 = keys2;
  c2h::host_vector<int> reference;
  Op::reference(h1.begin(), h1.end(), h2.begin(), h2.end(), std::back_inserter(reference));

  const int num = num_selected[0];
  REQUIRE(num == static_cast<int>(reference.size()));

  c2h::host_vector<int> keys_out_h(keys_out);
  keys_out_h.resize(num);
  REQUIRE(reference == keys_out_h);

  c2h::host_vector<int> values_out_h(values_out);
  values_out_h.resize(num);
  for (int i = 0; i < num; ++i)
  {
    const int key = keys_out_h[i];
    CAPTURE(i, key, values_out_h[i]);
    REQUIRE((values_out_h[i] == key * 2 || values_out_h[i] == key * 2 + 1));
    if (Op::values_from_first_input_only)
    {
      REQUIRE(values_out_h[i] == key * 2);
    }
  }
}

#if TEST_LAUNCH == 0
CUB_TEST("DeviceSetOps pairs work with default environment", "[set_ops][device]", CUB_SMALL, key_ops)
{
  using op      = c2h::get<0, TestType>;
  auto keys1    = c2h::device_vector<int>{0, 2, 2, 5};
  auto keys2    = c2h::device_vector<int>{0, 3, 3, 4};
  auto values1  = values_for(keys1, 0);
  auto values2  = values_for(keys2, 1);
  auto keys_out = c2h::device_vector<int>(keys1.size() + keys2.size(), thrust::default_init);
  auto vals_out = c2h::device_vector<int>(keys1.size() + keys2.size(), thrust::default_init);
  auto num      = c2h::device_vector<int>(1, thrust::default_init);

  REQUIRE(
    cudaSuccess
    == op::api(
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      keys_out.begin(),
      vals_out.begin(),
      num.begin()));
  check_pairs<op>(keys1, keys2, keys_out, vals_out, num);
}
#endif // TEST_LAUNCH == 0

CUB_TEST("DeviceSetOps pairs use environment", "[set_ops][device]", CUB_SMALL, key_ops)
{
  using op      = c2h::get<0, TestType>;
  auto keys1    = c2h::device_vector<int>{0, 1, 2, 2, 5, 7, 9};
  auto keys2    = c2h::device_vector<int>{0, 2, 3, 3, 4, 9};
  auto values1  = values_for(keys1, 0);
  auto values2  = values_for(keys2, 1);
  auto keys_out = c2h::device_vector<int>(keys1.size() + keys2.size(), thrust::default_init);
  auto vals_out = c2h::device_vector<int>(keys1.size() + keys2.size(), thrust::default_init);
  auto num      = c2h::device_vector<int>(1, thrust::default_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == op::api(
      nullptr,
      expected_bytes_allocated,
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      keys_out.begin(),
      vals_out.begin(),
      num.begin()));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};
  op::wrapper(
    keys1.begin(),
    values1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    values2.begin(),
    static_cast<int>(keys2.size()),
    keys_out.begin(),
    vals_out.begin(),
    num.begin(),
    ::cuda::std::less<>{},
    env);
  check_pairs<op>(keys1, keys2, keys_out, vals_out, num);
}

CUB_TEST("DeviceSetOps pairs use custom stream", "[set_ops][device]", CUB_SMALL, key_ops)
{
  using op      = c2h::get<0, TestType>;
  auto keys1    = c2h::device_vector<int>{0, 1, 2, 2, 5, 7, 9};
  auto keys2    = c2h::device_vector<int>{0, 2, 3, 3, 4, 9};
  auto values1  = values_for(keys1, 0);
  auto values2  = values_for(keys2, 1);
  auto keys_out = c2h::device_vector<int>(keys1.size() + keys2.size(), thrust::default_init);
  auto vals_out = c2h::device_vector<int>(keys1.size() + keys2.size(), thrust::default_init);
  auto num      = c2h::device_vector<int>(1, thrust::default_init);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == op::api(
      nullptr,
      expected_bytes_allocated,
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      keys_out.begin(),
      vals_out.begin(),
      num.begin()));

  auto stream_prop = stdexec::prop{::cuda::get_stream_t{}, ::cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};
  op::wrapper(
    keys1.begin(),
    values1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    values2.begin(),
    static_cast<int>(keys2.size()),
    keys_out.begin(),
    vals_out.begin(),
    num.begin(),
    ::cuda::std::less<>{},
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));
  check_pairs<op>(keys1, keys2, keys_out, vals_out, num);
  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

// See the keys env test for why tuned block sizes must be >= the fixed 256-thread partition kernel.
template <int ThreadsPerBlock>
struct set_ops_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const -> cub::SetOpsPolicy
  {
    return {ThreadsPerBlock, 1, cub::LOAD_DEFAULT, cub::BLOCK_SCAN_WARP_SCANS};
  }
};

using tuned_block_sizes =
  c2h::type_list<::cuda::std::integral_constant<unsigned int, 256>, ::cuda::std::integral_constant<unsigned int, 512>>;

CUB_TEST("DeviceSetOps pairs can be tuned", "[set_ops][device]", CUB_SMALL, key_ops, tuned_block_sizes)
{
  using op                                 = c2h::get<0, TestType>;
  constexpr unsigned int target_block_size = c2h::get<1, TestType>::value;

  auto keys1        = c2h::device_vector<int>{0, 2, 2, 5};
  auto keys2        = c2h::device_vector<int>{0, 3, 3, 4};
  auto values1      = values_for(keys1, 0);
  auto values2      = values_for(keys2, 1);
  auto keys_out     = c2h::device_vector<int>(keys1.size() + keys2.size(), thrust::default_init);
  auto vals_out     = c2h::device_vector<int>(keys1.size() + keys2.size(), thrust::default_init);
  auto num          = c2h::device_vector<int>(1, thrust::default_init);
  auto d_block_size = c2h::device_vector<unsigned int>(1, thrust::default_init);

  const block_size_extracting_op<::cuda::std::less<>> block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  auto env = ::cuda::execution::tune(set_ops_tuning<target_block_size>{});

  REQUIRE(
    cudaSuccess
    == op::api(
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      keys_out.begin(),
      vals_out.begin(),
      num.begin(),
      block_size_check,
      env));

  check_pairs<op>(keys1, keys2, keys_out, vals_out, num);
  REQUIRE(d_block_size[0] == target_block_size);
}
