// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_shuffle_sort.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/functional>

#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

namespace
{
constexpr int iterations = 128;

struct benchmark_op_t
{
  template <typename Key, int Items>
  __device__ __forceinline__ void operator()(Key (&keys)[Items]) const
  {
    cub::detail::WarpShuffleSort<Key, Items>{}.Sort(keys, cuda::std::less<>{});
  }
};

template <typename Key, int Items>
void full(nvbench::state& state, nvbench::type_list<Key, nvbench::enum_type<Items>>)
{
  const auto& kernel = ::benchmark_kernel<Items, Key, void, benchmark_op_t>;

  const bool latency = state.get_int64("latency") != 0;
  const int threads  = latency ? 32 : 128;
  const int blocks   = latency ? 1 : 32 * state.get_device().value().get_number_of_sms();

  state.add_element_count(static_cast<size_t>(blocks) * threads * Items * iterations);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    kernel<<<blocks, threads, 0, launch.get_stream()>>>(iterations, benchmark_op_t{});
  });
}

using key_types = nvbench::type_list<uint32_t, uint64_t>;
using items     = nvbench::enum_type_list<1, 2, 4, 8>;

NVBENCH_BENCH_TYPES(full, NVBENCH_TYPE_AXES(key_types, items))
  .set_type_axes_names({"KeyT", "items_per_lane"})
  .add_int64_axis("latency", {0, 1});
} // namespace
