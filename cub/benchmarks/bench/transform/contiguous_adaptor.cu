// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_transform.cuh>

#include <cuda/iterator>
#include <cuda/std/functional>

#include <nvbench_helper.cuh>

// Same-type transform over a device pointer. The stored values are contiguous; the iterator itself is not.
template <typename T>
struct plus_one_t
{
  __host__ __device__ T operator()(T value) const
  {
    return static_cast<T>(value + T{1});
  }
};

template <typename T>
void copy(nvbench::state& state, nvbench::type_list<T>)
try
{
  const auto elements     = state.get_int64("Elements{io}");
  const auto source       = state.get_string("Source");
  const bool as_transform = source == "transform";

  thrust::device_vector<T> in  = generate(static_cast<std::size_t>(elements));
  thrust::device_vector<T> out = generate(static_cast<std::size_t>(elements));

  auto* d_in  = thrust::raw_pointer_cast(in.data());
  auto* d_out = thrust::raw_pointer_cast(out.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cuda::std::execution::env{cuda::stream_ref{launch.get_stream().get_stream()}};
    if (as_transform)
    {
      auto it = cuda::transform_iterator(d_in, plus_one_t<T>{});
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceTransform::Transform,
        "Transform failed",
        cuda::std::make_tuple(it),
        d_out,
        static_cast<std::int32_t>(elements),
        cuda::std::identity{},
        env);
    }
    else
    {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceTransform::Transform,
        "Transform failed",
        cuda::std::make_tuple(d_in),
        d_out,
        static_cast<std::int32_t>(elements),
        plus_one_t<T>{},
        env);
    }
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

using value_types = nvbench::type_list<std::int8_t, std::int16_t, std::int32_t, std::int64_t, int128_t>;

NVBENCH_BENCH_TYPES(copy, NVBENCH_TYPE_AXES(value_types))
  .set_name("contiguous_adaptor")
  .set_type_axes_names({"T{ct}"})
  .add_string_axis("Source", {"pointer", "transform"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(24, 28, 4));
