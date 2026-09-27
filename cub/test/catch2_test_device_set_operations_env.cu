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

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSetOps::SetDifference, set_difference);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSetOps::SetIntersection, set_intersection);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSetOps::SetSymmetricDifference, set_symmetric_difference);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSetOps::SetUnion, set_union);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

// One tag per keys-only set operation, bundling the launch wrapper (single-phase env API), the explicit-temp-storage
// API (used to query the temporary-storage size), and the std reference algorithm.
struct op_difference
{
  template <typename... Args>
  static void wrapper(Args&&... args)
  {
    set_difference(::cuda::std::forward<Args>(args)...);
  }
  template <typename... Args>
  static cudaError_t api(Args&&... args)
  {
    return cub::DeviceSetOps::SetDifference(::cuda::std::forward<Args>(args)...);
  }
  template <typename It, typename Out, typename CompareOp>
  static void reference(It a1, It a2, It b1, It b2, Out out, CompareOp cmp)
  {
    std::set_difference(a1, a2, b1, b2, out, cmp);
  }
};
struct op_intersection
{
  template <typename... Args>
  static void wrapper(Args&&... args)
  {
    set_intersection(::cuda::std::forward<Args>(args)...);
  }
  template <typename... Args>
  static cudaError_t api(Args&&... args)
  {
    return cub::DeviceSetOps::SetIntersection(::cuda::std::forward<Args>(args)...);
  }
  template <typename It, typename Out, typename CompareOp>
  static void reference(It a1, It a2, It b1, It b2, Out out, CompareOp cmp)
  {
    std::set_intersection(a1, a2, b1, b2, out, cmp);
  }
};
struct op_symmetric_difference
{
  template <typename... Args>
  static void wrapper(Args&&... args)
  {
    set_symmetric_difference(::cuda::std::forward<Args>(args)...);
  }
  template <typename... Args>
  static cudaError_t api(Args&&... args)
  {
    return cub::DeviceSetOps::SetSymmetricDifference(::cuda::std::forward<Args>(args)...);
  }
  template <typename It, typename Out, typename CompareOp>
  static void reference(It a1, It a2, It b1, It b2, Out out, CompareOp cmp)
  {
    std::set_symmetric_difference(a1, a2, b1, b2, out, cmp);
  }
};
struct op_union
{
  template <typename... Args>
  static void wrapper(Args&&... args)
  {
    set_union(::cuda::std::forward<Args>(args)...);
  }
  template <typename... Args>
  static cudaError_t api(Args&&... args)
  {
    return cub::DeviceSetOps::SetUnion(::cuda::std::forward<Args>(args)...);
  }
  template <typename It, typename Out, typename CompareOp>
  static void reference(It a1, It a2, It b1, It b2, Out out, CompareOp cmp)
  {
    std::set_union(a1, a2, b1, b2, out, cmp);
  }
};

using key_ops = c2h::type_list<op_difference, op_intersection, op_symmetric_difference, op_union>;

template <typename Op, typename CompareOp>
void check_keys(const c2h::device_vector<int>& keys1,
                const c2h::device_vector<int>& keys2,
                const c2h::device_vector<int>& result,
                const c2h::device_vector<int>& num_selected,
                CompareOp cmp)
{
  c2h::host_vector<int> h1 = keys1;
  c2h::host_vector<int> h2 = keys2;
  c2h::host_vector<int> reference;
  Op::reference(h1.begin(), h1.end(), h2.begin(), h2.end(), std::back_inserter(reference), cmp);

  const int num = num_selected[0];
  REQUIRE(num == static_cast<int>(reference.size()));
  c2h::host_vector<int> result_h(result);
  result_h.resize(num);
  REQUIRE(reference == result_h);
}

#if TEST_LAUNCH == 0
CUB_TEST("DeviceSetOps keys work with default environment", "[set_ops][device]", CUB_SMALL, key_ops)
{
  using op    = c2h::get<0, TestType>;
  auto keys1  = c2h::device_vector<int>{0, 2, 2, 5};
  auto keys2  = c2h::device_vector<int>{0, 3, 3, 4};
  auto result = c2h::device_vector<int>(keys1.size() + keys2.size(), thrust::default_init);
  auto num    = c2h::device_vector<int>(1, thrust::default_init);

  // Direct call to the single-phase API with no environment argument.
  REQUIRE(cudaSuccess
          == op::api(keys1.begin(),
                     static_cast<int>(keys1.size()),
                     keys2.begin(),
                     static_cast<int>(keys2.size()),
                     result.begin(),
                     num.begin()));
  check_keys<op>(keys1, keys2, result, num, ::cuda::std::less<int>{});
}
#endif // TEST_LAUNCH == 0

CUB_TEST("DeviceSetOps keys use environment", "[set_ops][device]", CUB_SMALL, key_ops)
{
  using op    = c2h::get<0, TestType>;
  auto keys1  = c2h::device_vector<int>{0, 1, 2, 2, 5, 7, 9};
  auto keys2  = c2h::device_vector<int>{0, 2, 3, 3, 4, 9};
  auto result = c2h::device_vector<int>(keys1.size() + keys2.size(), thrust::default_init);
  auto num    = c2h::device_vector<int>(1, thrust::default_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == op::api(nullptr,
               expected_bytes_allocated,
               keys1.begin(),
               static_cast<int>(keys1.size()),
               keys2.begin(),
               static_cast<int>(keys2.size()),
               result.begin(),
               num.begin()));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};
  op::wrapper(
    keys1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    static_cast<int>(keys2.size()),
    result.begin(),
    num.begin(),
    ::cuda::std::less<>{},
    env);
  check_keys<op>(keys1, keys2, result, num, ::cuda::std::less<int>{});
}

CUB_TEST("DeviceSetOps keys use custom stream", "[set_ops][device]", CUB_SMALL, key_ops)
{
  using op    = c2h::get<0, TestType>;
  auto keys1  = c2h::device_vector<int>{0, 1, 2, 2, 5, 7, 9};
  auto keys2  = c2h::device_vector<int>{0, 2, 3, 3, 4, 9};
  auto result = c2h::device_vector<int>(keys1.size() + keys2.size(), thrust::default_init);
  auto num    = c2h::device_vector<int>(1, thrust::default_init);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == op::api(nullptr,
               expected_bytes_allocated,
               keys1.begin(),
               static_cast<int>(keys1.size()),
               keys2.begin(),
               static_cast<int>(keys2.size()),
               result.begin(),
               num.begin()));

  auto stream_prop = stdexec::prop{::cuda::get_stream_t{}, ::cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};
  op::wrapper(
    keys1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    static_cast<int>(keys2.size()),
    result.begin(),
    num.begin(),
    ::cuda::std::less<>{},
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));
  check_keys<op>(keys1, keys2, result, num, ::cuda::std::less<int>{});
  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

CUB_TEST(
  "DeviceSetOps keys respect a custom comparator through the environment", "[set_ops][device]", CUB_SMALL, key_ops)
{
  using op       = c2h::get<0, TestType>;
  const auto cmp = ::cuda::std::greater<int>{};
  // descending-sorted inputs
  auto keys1  = c2h::device_vector<int>{9, 7, 5, 2, 2, 1, 0};
  auto keys2  = c2h::device_vector<int>{9, 4, 3, 3, 2, 0};
  auto result = c2h::device_vector<int>(keys1.size() + keys2.size(), thrust::default_init);
  auto num    = c2h::device_vector<int>(1, thrust::default_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == op::api(
      nullptr,
      expected_bytes_allocated,
      keys1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      static_cast<int>(keys2.size()),
      result.begin(),
      num.begin(),
      cmp));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};
  op::wrapper(
    keys1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    static_cast<int>(keys2.size()),
    result.begin(),
    num.begin(),
    cmp,
    env);
  check_keys<op>(keys1, keys2, result, num, cmp);
}

// The tuning environment must change the block size used by the sweep kernel. Because the duplicate-aware merge path
// invokes the comparator inside the fixed-size (256-thread) partition kernel as well, the recorded block size is the
// maximum of the partition and sweep block sizes; we therefore tune to values >= 256 so the sweep block size dominates.
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

CUB_TEST("DeviceSetOps keys can be tuned", "[set_ops][device]", CUB_SMALL, key_ops, tuned_block_sizes)
{
  using op                                 = c2h::get<0, TestType>;
  constexpr unsigned int target_block_size = c2h::get<1, TestType>::value;

  auto keys1        = c2h::device_vector<int>{0, 2, 2, 5};
  auto keys2        = c2h::device_vector<int>{0, 3, 3, 4};
  auto result       = c2h::device_vector<int>(keys1.size() + keys2.size(), thrust::default_init);
  auto num          = c2h::device_vector<int>(1, thrust::default_init);
  auto d_block_size = c2h::device_vector<unsigned int>(1, thrust::default_init);

  const block_size_extracting_op<::cuda::std::less<>> block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  auto env = ::cuda::execution::tune(set_ops_tuning<target_block_size>{});

  REQUIRE(
    cudaSuccess
    == op::api(keys1.begin(),
               static_cast<int>(keys1.size()),
               keys2.begin(),
               static_cast<int>(keys2.size()),
               result.begin(),
               num.begin(),
               block_size_check,
               env));

  check_keys<op>(keys1, keys2, result, num, ::cuda::std::less<int>{});
  REQUIRE(d_block_size[0] == target_block_size);
}
