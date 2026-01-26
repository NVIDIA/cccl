// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_batch_memcpy.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the batch memcpy dispatcher after publishing the tuning API

template <class BufferOffsetT, class BlockOffsetT>
struct my_policy_hub
{
  static constexpr uint32_t BLOCK_THREADS         = 128U;
  static constexpr uint32_t BUFFERS_PER_THREAD    = 4U;
  static constexpr uint32_t TLEV_BYTES_PER_THREAD = 8U;

  static constexpr uint32_t LARGE_BUFFER_BLOCK_THREADS    = 256U;
  static constexpr uint32_t LARGE_BUFFER_BYTES_PER_THREAD = 32U;

  static constexpr uint32_t WARP_LEVEL_THRESHOLD  = 128;
  static constexpr uint32_t BLOCK_LEVEL_THRESHOLD = 8 * 1024;

  using buff_delay_constructor_t  = cub::detail::default_delay_constructor_t<BufferOffsetT>;
  using block_delay_constructor_t = cub::detail::default_delay_constructor_t<BlockOffsetT>;

  // from Policy500 of the CUB batch memcpy tunings
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    using AgentSmallBufferPolicyT = cub::detail::batch_memcpy::AgentBatchMemcpyPolicy<
      BLOCK_THREADS,
      BUFFERS_PER_THREAD,
      TLEV_BYTES_PER_THREAD,
      /* PREFER_POW2_BITS */ true,
      LARGE_BUFFER_BLOCK_THREADS * LARGE_BUFFER_BYTES_PER_THREAD,
      WARP_LEVEL_THRESHOLD,
      BLOCK_LEVEL_THRESHOLD,
      buff_delay_constructor_t,
      block_delay_constructor_t>;

    using AgentLargeBufferPolicyT =
      cub::detail::batch_memcpy::agent_large_buffer_policy<LARGE_BUFFER_BLOCK_THREADS, LARGE_BUFFER_BYTES_PER_THREAD>;
  };
};

C2H_TEST("DispatchBatchMemcpy::Dispatch: custom policy hub", "[device][memcpy]")
{
  using value_t         = cuda::std::uint8_t;
  using buffer_size_t   = cuda::std::uint32_t;
  using block_offset_t  = cuda::std::uint32_t;
  using buffer_offset_t = cub::detail::batch_memcpy::per_invocation_buffer_offset_t;

  const cuda::std::array<buffer_size_t, 5> buffer_sizes{3, 128, 512, 4096, 9000};

  c2h::host_vector<c2h::device_vector<value_t>> in_buffers(buffer_sizes.size());
  c2h::host_vector<c2h::device_vector<value_t>> out_buffers(buffer_sizes.size());

  c2h::host_vector<value_t*> h_in_ptrs(buffer_sizes.size());
  c2h::host_vector<value_t*> h_out_ptrs(buffer_sizes.size());
  c2h::host_vector<buffer_size_t> h_sizes(buffer_sizes.size());

  for (buffer_size_t i = 0; i < buffer_sizes.size(); ++i)
  {
    const auto bytes = buffer_sizes[i];
    in_buffers[i].resize(bytes);
    out_buffers[i].resize(bytes);
    c2h::gen(C2H_SEED(1), in_buffers[i]);

    h_in_ptrs[i]  = thrust::raw_pointer_cast(in_buffers[i].data());
    h_out_ptrs[i] = thrust::raw_pointer_cast(out_buffers[i].data());
    h_sizes[i]    = bytes;
  }

  c2h::device_vector<value_t*> d_in_ptrs    = h_in_ptrs;
  c2h::device_vector<value_t*> d_out_ptrs   = h_out_ptrs;
  c2h::device_vector<buffer_size_t> d_sizes = h_sizes;

  using policy_hub_t = my_policy_hub<buffer_offset_t, block_offset_t>;
  using dispatch_t =
    cub::detail::DispatchBatchMemcpy<value_t**, value_t**, buffer_size_t*, block_offset_t, CopyAlg::Memcpy, policy_hub_t>;

  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(d_in_ptrs.data()),
    thrust::raw_pointer_cast(d_out_ptrs.data()),
    thrust::raw_pointer_cast(d_sizes.data()),
    static_cast<cuda::std::int64_t>(buffer_sizes.size()),
    /* stream */ nullptr);
  c2h::device_vector<::cuda::std::uint8_t> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    thrust::raw_pointer_cast(d_in_ptrs.data()),
    thrust::raw_pointer_cast(d_out_ptrs.data()),
    thrust::raw_pointer_cast(d_sizes.data()),
    static_cast<cuda::std::int64_t>(buffer_sizes.size()),
    /* stream */ nullptr);

  for (size_t i = 0; i < buffer_sizes.size(); ++i)
  {
    c2h::host_vector<value_t> host_in(in_buffers[i]);
    c2h::host_vector<value_t> host_out(out_buffers[i]);
    REQUIRE(host_out == host_in);
  }
}
