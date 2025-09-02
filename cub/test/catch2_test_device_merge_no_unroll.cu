// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Define this to enable compile-time optimizations that avoid unrolling
// some loops in the merge kernels. This reduces the compile time for this
// test from ~8 minutes to >2.
#define CCCL_AVOID_SORT_UNROLL

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include <algorithm>

#include <test_util.h>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceMerge::MergePairs, merge_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMerge::MergeKeys, merge_keys);

template <typename Key,
          typename Offset,
          typename CompareOp = cuda::std::less<Key>,
          typename MergeKeys = decltype(::merge_keys)>
void test_keys(Offset size1 = 3623, Offset size2 = 6346, CompareOp compare_op = {}, MergeKeys merge_keys = ::merge_keys)
{
  CAPTURE(c2h::type_name<Key>(), c2h::type_name<Offset>(), size1, size2);

  c2h::device_vector<Key> keys1_d(size1);
  c2h::device_vector<Key> keys2_d(size2);

  c2h::gen(C2H_SEED(1), keys1_d);
  c2h::gen(C2H_SEED(1), keys2_d);

  thrust::sort(c2h::device_policy, keys1_d.begin(), keys1_d.end(), compare_op);
  thrust::sort(c2h::device_policy, keys2_d.begin(), keys2_d.end(), compare_op);
  // CAPTURE(keys1_d, keys2_d);

  c2h::device_vector<Key> result_d(size1 + size2);
  merge_keys(thrust::raw_pointer_cast(keys1_d.data()),
             static_cast<Offset>(keys1_d.size()),
             thrust::raw_pointer_cast(keys2_d.data()),
             static_cast<Offset>(keys2_d.size()),
             thrust::raw_pointer_cast(result_d.data()),
             compare_op);

  c2h::host_vector<Key> keys1_h = keys1_d;
  c2h::host_vector<Key> keys2_h = keys2_d;
  c2h::host_vector<Key> reference_h(size1 + size2);
  std::merge(keys1_h.begin(), keys1_h.end(), keys2_h.begin(), keys2_h.end(), reference_h.begin(), compare_op);

  // FIXME(bgruber): comparing std::vectors (slower than thrust vectors) but compiles a lot faster
  CHECK((detail::to_vec(reference_h) == detail::to_vec(c2h::host_vector<Key>(result_d))));
}

using large_type_fallb = c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<56>::type>;
using large_type_vsmem = c2h::custom_type_t<c2h::equal_comparable_t, c2h::less_comparable_t, c2h::huge_data<774>::type>;

struct fallback_test_policy_hub
{
  struct max_policy : cub::ChainedPolicy<100, max_policy, max_policy>
  {
    using merge_policy = cub::detail::merge::
      agent_policy_t<128, 7, cub::BLOCK_LOAD_WARP_TRANSPOSE, cub::LOAD_DEFAULT, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };
};

C2H_TEST("DeviceMerge::MergeKeys large key types", "[merge][device]", c2h::type_list<large_type_vsmem, large_type_fallb>)
{
  using key_t    = c2h::get<0, TestType>;
  using offset_t = int;

  constexpr auto agent_sm = sizeof(key_t) * 128 * 7;
  constexpr auto fallback_sm =
    sizeof(key_t) * cub::detail::merge::fallback_BLOCK_THREADS * cub::detail::merge::fallback_ITEMS_PER_THREAD;
  static_assert(agent_sm > cub::detail::max_smem_per_block,
                "key_t is not big enough to exceed SM and trigger fallback policy");
  static_assert(cuda::std::is_same_v<key_t, large_type_fallb> == (fallback_sm <= cub::detail::max_smem_per_block),
                "SM consumption by fallback policy should fit into max_smem_per_block");

  test_keys<key_t, offset_t>(
    3623,
    6346,
    cuda::std::less<key_t>{},
    [](const key_t* k1, offset_t s1, const key_t* k2, offset_t s2, key_t* r, cuda::std::less<key_t> co) {
      using dispatch_t = cub::detail::merge::dispatch_t<
        const key_t*,
        const cub::NullType*,
        const key_t*,
        const cub::NullType*,
        key_t*,
        cub::NullType*,
        offset_t,
        cuda::std::less<key_t>,
        fallback_test_policy_hub>; // use a fixed policy for this test so the needed shared memory is deterministic

      std::size_t temp_storage_bytes = 0;
      dispatch_t::dispatch(
        nullptr, temp_storage_bytes, k1, nullptr, s1, k2, nullptr, s2, r, nullptr, co, cudaStream_t{0});

      c2h::device_vector<char> temp_storage(temp_storage_bytes);
      dispatch_t::dispatch(
        thrust::raw_pointer_cast(temp_storage.data()),
        temp_storage_bytes,
        k1,
        nullptr,
        s1,
        k2,
        nullptr,
        s2,
        r,
        nullptr,
        co,
        cudaStream_t{0});
    });
}
