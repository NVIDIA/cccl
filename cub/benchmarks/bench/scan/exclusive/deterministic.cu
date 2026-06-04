// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_scan.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/std/__functional/invoke.h>

#include <nvbench_helper.cuh>

template <typename T, typename OffsetT>
static void exclusive_scan(nvbench::state& state, nvbench::type_list<T, OffsetT>)
try
{
  using init_t    = T;
  using offset_t  = OffsetT;
  using scan_op_t = ::cuda::std::plus<T>;

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements, thrust::no_init);

  const T* d_input = thrust::raw_pointer_cast(input.data());
  T* d_output      = thrust::raw_pointer_cast(output.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch, cuda::execution::require(cuda::execution::determinism::run_to_run));
    _CCCL_TRY_CUDA_API(
      cub::DeviceScan::ExclusiveScan,
      "ExclusiveScan failed",
      d_input,
      d_output,
      scan_op_t{},
      init_t{},
      static_cast<offset_t>(elements),
      env);
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

using types   = nvbench::type_list<float, double>;
using offsets = nvbench::type_list<int64_t>;

NVBENCH_BENCH_TYPES(exclusive_scan, NVBENCH_TYPE_AXES(types, offsets))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
