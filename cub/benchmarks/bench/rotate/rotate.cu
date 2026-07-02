// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_rotate.cuh>

#include <nvbench_helper.cuh>

template <bool AsPercentage, typename T>
void rotate_benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  const auto num_bytes    = static_cast<size_t>(state.get_int64("Bytes{io}"));
  const auto num_elements = num_bytes / sizeof(T);

  size_t rot_dist;
  if constexpr (AsPercentage)
  {
    const auto rot_pct = state.get_float64("RotatePercentage");
    rot_dist           = rot_pct == 0.0 ? size_t{1} : static_cast<size_t>(rot_pct * num_elements);
  }
  else
  {
    rot_dist = static_cast<size_t>(state.get_int64("RotateDistance"));
  }
  const auto num_unaligned_elems = static_cast<int>(state.get_int64("NumUnalignedElems"));

  if (rot_dist >= num_elements)
  {
    state.skip("Skipped: rotate distance >= array size");
    return;
  }
  if (num_unaligned_elems * sizeof(T) >= cub::detail::rotate::BYTES_PER_SECTOR)
  {
    state.skip("Skipped: unaligned elems exceed sector size.");
    return;
  }

  // Allocate with extra room for unaligned offset
  thrust::device_vector<T> data = generate(num_elements + num_unaligned_elems);
  T* d_data                     = thrust::raw_pointer_cast(data.data()) + num_unaligned_elems;

  // Query pass
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::RotateState_t rotate_state;
  cub::DeviceRotate::Rotate(d_temp_storage, temp_storage_bytes, rotate_state, d_data, num_elements, rot_dist);

  thrust::device_vector<nvbench::uint8_t> temp(temp_storage_bytes, thrust::no_init);
  d_temp_storage = thrust::raw_pointer_cast(temp.data());

  state.add_element_count(num_elements);
  state.add_global_memory_reads<T>(num_elements);
  state.add_global_memory_writes<T>(num_elements);

  const auto algo    = cub::detail::rotate::get_algorithm_to_use<T>(rot_dist, rotate_state.max_distance_, nullptr);
  auto& algo_summary = state.add_summary("Algorithm");
  algo_summary.set_string("name", "Algorithm");
  algo_summary.set_string(
    "value",
    algo == cub::detail::rotate::RotateAlgo::Short ? "short"
    : algo == cub::detail::rotate::RotateAlgo::Long
      ? "long"
      : "naive");

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceRotate::Rotate(
      d_temp_storage, temp_storage_bytes, rotate_state, d_data, num_elements, rot_dist, launch.get_stream());
  });
}

// NVBENCH_BENCH_TYPES instantiates its callable from the type axis alone, so wrap each distance
// mode in a single-type-parameter entry point that pins the `AsPercentage` template argument.
template <typename T>
void rotate_benchmark_distance(nvbench::state& state, nvbench::type_list<T> types)
{
  rotate_benchmark<false, T>(state, types);
}

template <typename T>
void rotate_benchmark_percentage(nvbench::state& state, nvbench::type_list<T> types)
{
  rotate_benchmark<true, T>(state, types);
}

using TypeList = nvbench::type_list<uint8_t, uint16_t, uint32_t, uint64_t>;

// Short kernel: absolute rotate distances that stay within the short-path tile threshold.
NVBENCH_BENCH_TYPES(rotate_benchmark_distance, NVBENCH_TYPE_AXES(TypeList))
  .set_name("rotate_short")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Bytes{io}", nvbench::range(26, 32, 2))
  .add_int64_axis("RotateDistance", {1, 100, 500, 5000, 10000, 20000, 30000, 32700})
  .add_int64_axis("NumUnalignedElems", {0LL, 1LL});

// Long/naive algorithm: rotate distance as a fraction of the array size.
NVBENCH_BENCH_TYPES(rotate_benchmark_percentage, NVBENCH_TYPE_AXES(TypeList))
  .set_name("rotate_long")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Bytes{io}", nvbench::range(26, 32, 2))
  .add_float64_axis("RotatePercentage", {0.01, 0.3, 0.6, 0.9})
  .add_int64_axis("NumUnalignedElems", {0LL, 1LL});
