// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_select_if.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/functional>

#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the select if dispatcher after publishing the tuning API

template <class InputT>
struct my_policy_hub
{
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    static constexpr int nominal_4b_items_per_thread = 10;
    static constexpr int items_per_thread =
      cuda::std::clamp(nominal_4b_items_per_thread * 4 / int{sizeof(InputT)}, 1, nominal_4b_items_per_thread);
    using SelectIfPolicyT =
      AgentSelectIfPolicy<128,
                          items_per_thread,
                          BLOCK_LOAD_DIRECT,
                          LOAD_CA,
                          BLOCK_SCAN_WARP_SCANS,
                          cub::detail::fixed_delay_constructor_t<350, 450>>;
  };
};

struct is_even_t
{
  __host__ __device__ bool operator()(int value) const
  {
    return (value & 1) == 0;
  }
};

C2H_TEST("DispatchSelectIf::Dispatch: custom policy hub", "[select_if][device]")
{
  using value_t  = int;
  using offset_t = int;

  const c2h::host_vector<value_t> h_in{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  c2h::device_vector<value_t> d_in = h_in;
  c2h::device_vector<value_t> d_out(h_in.size());
  c2h::device_vector<offset_t> d_num_selected(1, 0);

  c2h::host_vector<value_t> expected;
  expected.reserve(h_in.size());
  for (const auto value : h_in)
  {
    if (is_even_t{}(value))
    {
      expected.push_back(value);
    }
  }

  using policy_hub_t = my_policy_hub<value_t>;
  using dispatch_t =
    DispatchSelectIf<value_t*, NullType*, value_t*, offset_t*, is_even_t, NullType, offset_t, SelectImpl::Select, policy_hub_t>;

  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(d_in.data()),
    nullptr,
    thrust::raw_pointer_cast(d_out.data()),
    thrust::raw_pointer_cast(d_num_selected.data()),
    is_even_t{},
    NullType{},
    static_cast<offset_t>(h_in.size()),
    /* stream */ nullptr);
  c2h::device_vector<unsigned char> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    thrust::raw_pointer_cast(d_in.data()),
    nullptr,
    thrust::raw_pointer_cast(d_out.data()),
    thrust::raw_pointer_cast(d_num_selected.data()),
    is_even_t{},
    NullType{},
    static_cast<offset_t>(h_in.size()),
    /* stream */ nullptr);

  const auto num_selected         = static_cast<std::size_t>(d_num_selected[0]);
  c2h::host_vector<value_t> h_out = d_out;
  h_out.resize(num_selected);
  expected.resize(num_selected);
  REQUIRE(h_out == expected);
}
