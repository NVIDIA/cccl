// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Composes affine maps x -> a * x + b, an associative but not commutative operation. Each item holds two values of
// type T, so the items are twice the size of T.

#include <cub/device/device_reduce.cuh>

#include <thrust/transform.h>

#include <cuda/std/cstdint>

#include <nvbench_helper.cuh>

template <typename T>
struct affine_t
{
  T a;
  T b;
};

template <typename T>
struct compose_t
{
  __device__ affine_t<T> operator()(const affine_t<T>& lhs, const affine_t<T>& rhs) const
  {
    // Promote small types to unsigned int, so the products wrap around instead of overflowing int
    using wide_t = decltype(T{} * 1u);
    return {static_cast<T>(wide_t{rhs.a} * lhs.a), static_cast<T>(wide_t{rhs.a} * lhs.b + rhs.b)};
  }
};

template <typename T>
struct to_affine_t
{
  __device__ affine_t<T> operator()(T value) const
  {
    return {static_cast<T>(value | 1), static_cast<T>(value >> 1)};
  }
};

template <typename T, typename OffsetT>
void non_commutative(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using item_t = affine_t<T>;

  const auto elements = state.get_int64("Elements{io}");

  const thrust::device_vector<T> values = generate(elements);
  thrust::device_vector<item_t> in(elements, thrust::no_init);
  thrust::transform(values.begin(), values.end(), in.begin(), to_affine_t<T>{});
  thrust::device_vector<item_t> out(1, thrust::no_init);

  auto d_in  = thrust::raw_pointer_cast(in.data());
  auto d_out = thrust::raw_pointer_cast(out.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<item_t>(elements, "Size");
  state.add_global_memory_writes<item_t>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceReduce::ReduceNonCommutative,
      "ReduceNonCommutative failed",
      d_in,
      d_out,
      static_cast<OffsetT>(elements),
      compose_t<T>{},
      item_t{1, 0},
      cub_bench_env(alloc, launch));
  });
}

using value_types = nvbench::type_list<cuda::std::uint16_t, cuda::std::uint32_t, cuda::std::uint64_t>;

NVBENCH_BENCH_TYPES(non_commutative, NVBENCH_TYPE_AXES(value_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
