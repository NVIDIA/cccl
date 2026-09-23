// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_scan.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/argument>
#include <cuda/execution>
#include <cuda/std/functional>
#include <cuda/std/utility>

#include <cstddef>

#include <nvbench_helper.cuh>

#include <nvbench/range.cuh>
#include <nvbench/types.cuh>

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5
// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1

#define USES_LOOKAHEAD() 0
using op_t = cuda::std::plus<>;
#include "../policy_selector.h"

template <typename T, typename OffsetT>
void deferred_sum(nvbench::state& state, nvbench::type_list<T, OffsetT>)
try
{
  using init_value_t             = T;
  using accum_t [[maybe_unused]] = cuda::std::__accumulator_t<op_t, init_value_t, T>;

  const auto requested_elements = state.get_int64("Elements{io}");
  if (!cuda::std::in_range<OffsetT>(requested_elements))
  {
    state.skip("Skipping: Elements{io} is not representable by OffsetT.");
    return;
  }
  const auto elements = static_cast<std::size_t>(requested_elements);

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements);
  thrust::device_vector<OffsetT> device_num_items(1, static_cast<OffsetT>(elements));

  const auto d_input     = thrust::raw_pointer_cast(input.data());
  const auto d_output    = thrust::raw_pointer_cast(output.data());
  const auto d_num_items = thrust::raw_pointer_cast(device_num_items.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(policy_selector<accum_t>{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DeviceScan::ExclusiveSum, "ExclusiveSum failed", d_input, d_output, cuda::args::deferred{d_num_items}, env);
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

using value_types = all_types;

NVBENCH_BENCH_TYPES(deferred_sum, NVBENCH_TYPE_AXES(value_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 32, 4));
