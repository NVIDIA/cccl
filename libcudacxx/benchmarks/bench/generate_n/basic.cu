// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>

#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/stream_ref>

#include "nvbench_helper.cuh"

template <typename T>
struct generator
{
  _CCCL_DEVICE_API _CCCL_FORCEINLINE auto operator()() const -> T
  {
    return 42;
  }
};

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> output(elements, thrust::no_init);

  state.add_element_count(elements);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc{};
  auto policy = cuda::execution::__cub_par_unseq.with_memory_resource(alloc);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               cuda::std::generate_n(
                 policy.with_stream(launch.get_stream().get_stream()), output.begin(), elements, generator<T>{});
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
