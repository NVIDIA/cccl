// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_rotate.cuh>

#include <cstddef>
#include <cstdint>

#include <nvbench_helper.cuh>

template <typename T>
void bfs_visit_order_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  using namespace cub::detail::rotate;

  const auto num_bytes    = static_cast<size_t>(state.get_int64("Bytes{io}"));
  const auto num_elements = num_bytes / sizeof(T);
  const auto rot_pct      = state.get_float64("RotatePercentage");
  const auto rot_dist     = rot_pct == 0.0 ? size_t{1} : static_cast<size_t>(rot_pct * num_elements);
  const auto head_size    = static_cast<uint32_t>(state.get_int64("HeadSize"));

  if (rot_dist >= num_elements)
  {
    state.skip("Skipped: rotate distance >= array size");
    return;
  }

  constexpr auto sector_elements = BYTES_PER_SECTOR / sizeof(T);
  if (head_size >= sector_elements)
  {
    state.skip("Skipped: head size exceeds sector size.");
    return;
  }

  cuda::compute_capability cc{};
  NVBENCH_CUDA_CALL(cub::detail::ptx_compute_cap(cc));
  const auto long_policy = policy_selector{}(cc).long_algorithm;

  state.add_element_count(num_elements);
  state.exec(nvbench::exec_tag::no_gpu, [&](nvbench::launch&) {
    do_not_optimize(bfs_visit_order<T>(num_elements, rot_dist, head_size, long_policy));
  });
}

using TypeList = nvbench::type_list<uint8_t>;

NVBENCH_BENCH_TYPES(bfs_visit_order_benchmark, NVBENCH_TYPE_AXES(TypeList))
  .set_name("bfs_visit_order")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Bytes{io}", nvbench::range(30, 32, 2))
  .add_float64_axis("RotatePercentage", // Random nums to force dependency chains
                    {0.03366275, 0.05408525, 0.07465742, 0.17368588,
                     0.22188703, 0.24204597, 0.25078173, 0.29280947,
                     0.41545696, 0.43265761, 0.45944469, 0.49439498})
  .add_int64_axis("HeadSize", {0LL, 1LL})
  .set_is_cpu_only(true);
