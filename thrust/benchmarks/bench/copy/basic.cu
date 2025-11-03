// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <nvbench_helper.cuh>

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements, T{1});
  thrust::device_vector<T> output(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::copy(policy(alloc, launch), input.cbegin(), input.cend(), output.begin());
             });
}

// Non-trivially-copyable/relocatable type which is not allowed to be copied using std::memcpy or cudaMemcpy
struct non_trivial
{
  int a;
  int b;

  non_trivial() = default;

  _CCCL_HOST_DEVICE explicit non_trivial(int i)
      : a(i)
      , b(i)
  {}

  // the user-defined copy constructor prevents the type from being trivially copyable
  _CCCL_HOST_DEVICE non_trivial(const non_trivial& nt)
      : a(nt.a)
      , b(nt.b)
  {}
};

static_assert(!::cuda::std::is_trivially_copyable<non_trivial>::value, ""); // as required by the C++ standard
static_assert(!thrust::is_trivially_relocatable<non_trivial>::value, ""); // thrust uses this check internally

using types =
  nvbench::type_list<nvbench::uint8_t, nvbench::uint16_t, nvbench::uint32_t, nvbench::uint64_t, non_trivial>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
