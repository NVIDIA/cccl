//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cuda/memory_pool>
#include <cuda/std/execution>
#include <cuda/stream>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements         = static_cast<std::size_t>(state.get_int64("Elements"));
  const bit_entropy entropy   = str_to_entropy(state.get_string("Entropy"));
  thrust::device_vector<T> in = generate(elements, entropy);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc{};
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               cuda::std::sort(cuda_policy(alloc, launch), in.begin(), in.end());
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});

struct fake_less
{
  template <class T, class U>
  [[nodiscard]] _CCCL_API constexpr bool operator()(const T& t, const U& u) const
  {
    // complex is not less than comparable, so just compare the first element
    if constexpr (cuda::std::__is_cpp17_less_than_comparable_v<T, U>)
    {
      return t < u;
    }
    else
    {
      return cuda::std::get<0>(t) < cuda::std::get<0>(u);
    }
  }
};

template <typename T>
static void with_predicate(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements         = static_cast<std::size_t>(state.get_int64("Elements"));
  const bit_entropy entropy   = str_to_entropy(state.get_string("Entropy"));
  thrust::device_vector<T> in = generate(elements, entropy);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc{};
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               cuda::std::sort(cuda_policy(alloc, launch), in.begin(), in.end(), fake_less{});
             });
}

NVBENCH_BENCH_TYPES(with_predicate, NVBENCH_TYPE_AXES(all_types))
  .set_name("with_predicate")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});
