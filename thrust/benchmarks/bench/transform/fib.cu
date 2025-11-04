// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <nvbench_helper.cuh>

template <class InT, class OutT>
struct fib_t
{
  __device__ OutT operator()(InT n)
  {
    OutT t1 = 0;
    OutT t2 = 1;

    if (n < 1)
    {
      return t1;
    }
    else if (n == 1)
    {
      return t1;
    }
    else if (n == 2)
    {
      return t2;
    }
    for (InT i = 3; i <= n; ++i)
    {
      const auto next = t1 + t2;
      t1              = t2;
      t2              = next;
    }

    return t2;
  }
};

template <typename T>
static void fib(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input = generate(elements, bit_entropy::_1_000, T{0}, T{42});
  thrust::device_vector<T> output(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<nvbench::uint32_t>(elements);

  fib_t<T, nvbench::uint32_t> op{};
  caching_allocator_t alloc; // transform shouldn't allocate, but let's be consistent
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::transform(policy(alloc, launch), input.cbegin(), input.cend(), output.begin(), op);
             });
}

using types = nvbench::type_list<nvbench::uint32_t, nvbench::uint64_t>;

NVBENCH_BENCH_TYPES(fib, NVBENCH_TYPE_AXES(types))
  .set_name("fib")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
