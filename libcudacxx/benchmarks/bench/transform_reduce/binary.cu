//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>

#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/stream>

#include "nvbench_helper.cuh"

template <typename T>
static void binary(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> in = generate(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(1);

  caching_allocator_t alloc{};

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    do_not_optimize(cuda::std::transform_reduce(
      cuda_policy(alloc, launch),
      in.begin(),
      in.end(),
      cuda::constant_iterator<int>{42},
      42,
      cuda::std::plus<T>{},
      cuda::std::multiplies<T>{}));
  });
}

NVBENCH_BENCH_TYPES(binary, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
