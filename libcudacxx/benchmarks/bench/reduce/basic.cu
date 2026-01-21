// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>

#include <cuda/memory_pool>
#include <cuda/std/__pstl/reduce.h>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> in = generate(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(1);

  ::cuda::stream stream{::cuda::device_ref{0}};
  ::cuda::device_memory_pool_ref alloc = ::cuda::device_default_memory_pool(stream.device());

  auto policy = cuda::execution::__cub_par_unseq.set_stream(stream).set_memory_resource(alloc);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    do_not_optimize(cuda::std::reduce(policy, in.begin(), in.end()));
  });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
