// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cuda/__cccl_config>

#if _CCCL_PP_COUNT(__CUDA_ARCH_LIST__) != 1
#  warning "This benchmark does not support being compiled for multiple architectures. Disabling it."
#else // _CCCL_PP_COUNT(__CUDA_ARCH_LIST__) != 1

#  include <cub/device/dispatch/dispatch_scan.cuh>

#  include <nvbench_helper.cuh>

template <typename T, typename OffsetT>
static void exclusive_scan(nvbench::state& state, nvbench::type_list<T, OffsetT>)
try
{
  using input_it_t  = const T*;
  using output_it_t = T*;
  using offset_t    = cub::detail::choose_offset_t<OffsetT>;
  using scan_op_t   = ::cuda::std::plus<T>;
  using init_t      = cub::detail::InputValue<T>;

  const auto elements                 = static_cast<::cuda::std::size_t>(state.get_int64("Elements{io}"));
  const bool run_to_run_deterministic = state.get_int64("Det") != 0;

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements);

  T* d_input  = thrust::raw_pointer_cast(input.data());
  T* d_output = thrust::raw_pointer_cast(output.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  size_t tmp_size;
  if (run_to_run_deterministic)
  {
    cub::detail::scan::dispatch_with_accum<T, cub::ForceInclusive::No, true>(
      nullptr, tmp_size, d_input, d_output, scan_op_t{}, init_t{T{}}, static_cast<offset_t>(elements), 0 /* stream */);
  }
  else
  {
    cub::detail::scan::dispatch_with_accum<T, cub::ForceInclusive::No, false>(
      nullptr, tmp_size, d_input, d_output, scan_op_t{}, init_t{T{}}, static_cast<offset_t>(elements), 0 /* stream */);
  }

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    if (run_to_run_deterministic)
    {
      cub::detail::scan::dispatch_with_accum<T, cub::ForceInclusive::No, true>(
        thrust::raw_pointer_cast(tmp.data()),
        tmp_size,
        d_input,
        d_output,
        scan_op_t{},
        init_t{T{}},
        static_cast<offset_t>(elements),
        launch.get_stream());
    }
    else
    {
      cub::detail::scan::dispatch_with_accum<T, cub::ForceInclusive::No, false>(
        thrust::raw_pointer_cast(tmp.data()),
        tmp_size,
        d_input,
        d_output,
        scan_op_t{},
        init_t{T{}},
        static_cast<offset_t>(elements),
        launch.get_stream());
    }
  });
}
catch (const std::exception& e)
{
  state.skip(e.what());
}

using types   = nvbench::type_list<float, double>;
using offsets = nvbench::type_list<int64_t>;

NVBENCH_BENCH_TYPES(exclusive_scan, NVBENCH_TYPE_AXES(types, offsets))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_axis("Det", {0, 1})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));

#endif // _CCCL_PP_COUNT(__CUDA_ARCH_LIST__) == 1
