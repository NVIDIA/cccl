// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <thrust/device_vector.h>
#include <thrust/uninitialized_copy.h>

#include <nvbench_helper.cuh>

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements, T{0xAA});
  thrust::device_vector<T> output(elements, thrust::default_init);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::uninitialized_copy(policy(alloc, launch), input.cbegin(), input.cend(), output.begin());
             });
}

// Not allowed to be copied using std::memcpy or cudaMemcpy. Cannot use TMA copies.
struct no_copy
{
  nvbench::uint32_t a;

  no_copy() = default;

  _CCCL_HOST_DEVICE no_copy(nvbench::uint32_t i)
      : a(i)
  {}

  // the user-defined copy constructor prevents the type from being trivially copyable
  _CCCL_HOST_DEVICE no_copy(const no_copy& nt)
      : a(nt.a)
  {}
};

static_assert(::cuda::std::is_trivially_default_constructible_v<no_copy>);
static_assert(!::cuda::std::is_trivially_copyable_v<no_copy>); // as required by the C++ standard
static_assert(!thrust::is_trivially_relocatable_v<no_copy>); // thrust uses this check internally

// Requires use of placement new
struct no_construct
{
  nvbench::uint32_t a = 1337;
};

static_assert(!::cuda::std::is_trivially_default_constructible_v<no_construct>);
static_assert(::cuda::std::is_trivially_copyable_v<no_construct>); // as required by the C++ standard
static_assert(thrust::is_trivially_relocatable_v<no_construct>); // thrust uses this check internally

using types =
  nvbench::type_list<nvbench::uint8_t, nvbench::uint16_t, nvbench::uint32_t, nvbench::uint64_t, no_copy, no_construct>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
