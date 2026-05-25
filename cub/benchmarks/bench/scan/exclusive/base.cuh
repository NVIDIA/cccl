// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/device/device_scan.cuh>

#include <cuda/std/__functional/invoke.h>

#include <nvbench_helper.cuh>

#include "../policy_selector.h"

template <typename T, typename OffsetT>
static void basic(nvbench::state& state, nvbench::type_list<T, OffsetT>)
try
{
  using init_t         = T;
  using wrapped_init_t = cub::detail::InputValue<init_t>;
  using accum_t        = ::cuda::std::__accumulator_t<op_t, init_t, T>;
  using input_it_t     = const T*;
  using output_it_t    = T*;
  using offset_t       = cub::detail::choose_offset_t<OffsetT>;
#if USES_WARPSPEED()
  static_assert(sizeof(offset_t) == sizeof(size_t)); // warpspeed scan uses size_t internally
#endif // USES_WARPSPEED()

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  if (sizeof(offset_t) == 4 && elements > std::numeric_limits<offset_t>::max())
  {
    state.skip("Skipping: input size exceeds 32-bit offset type capacity.");
    return;
  }

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements);

  const T* d_input = thrust::raw_pointer_cast(input.data());
  T* d_output      = thrust::raw_pointer_cast(output.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  size_t tmp_size;
  cub::detail::scan::dispatch_with_accum<accum_t>(
    nullptr,
    tmp_size,
    d_input,
    d_output,
    op_t{},
    wrapped_init_t{T{}},
    static_cast<offset_t>(input.size()),
    nullptr /* stream */
#if !TUNE_BASE
    ,
    policy_selector<accum_t>{}
#endif // !TUNE_BASE
  );

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::detail::scan::dispatch_with_accum<accum_t>(
      thrust::raw_pointer_cast(tmp.data()),
      tmp_size,
      d_input,
      d_output,
      op_t{},
      wrapped_init_t{T{}},
      static_cast<offset_t>(input.size()),
      launch.get_stream()
#if !TUNE_BASE
        ,
      policy_selector<accum_t>{}
#endif // !TUNE_BASE
    );
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(all_types, scan_offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 32, 4));
