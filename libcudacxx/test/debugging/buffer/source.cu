// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/array>
#include <cuda/stream>

#include <vector>

#include <cuda_runtime_api.h>

template <class T>
[[gnu::noinline]] void keep_for_debugger(const T& value)
{
  asm volatile("" : : "g"(&value) : "memory");
}

[[gnu::noinline]] void inspect_normal(const cuda::device_buffer<int>& values)
{
  keep_for_debugger(values);
}

using device_buffer_alias = cuda::buffer<int, cuda::mr::device_accessible>;

[[gnu::noinline]] void inspect_alias(const device_buffer_alias& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_vector(const std::vector<cuda::device_buffer<int>>& values)
{
  keep_for_debugger(values[0]);
  keep_for_debugger(values[1]);
}

template <class Buffer>
[[gnu::noinline]] void inspect_host_device(const Buffer& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_empty(const cuda::device_buffer<int>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::device_buffer<int>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_after_update(const cuda::device_buffer<int>& values)
{
  keep_for_debugger(values);
}

int main()
{
  constexpr cuda::device_ref device{0};
  cuda::stream stream{device};

  const cuda::std::array normal_host_values{-56, 22, 94, -13, 7, 41, -82, 0, 63, -5};
  const auto normal_values = cuda::make_device_buffer<int>(stream, device, normal_host_values);

  const cuda::std::array alias_host_values{17, -31, 8, 55};
  const device_buffer_alias aliased_values = cuda::make_device_buffer<int>(stream, device, alias_host_values);

  std::vector<cuda::device_buffer<int>> buffer_vector;
  buffer_vector.emplace_back(cuda::make_device_buffer<int>(stream, device, cuda::std::array{-2, 4, 6}));
  buffer_vector.emplace_back(cuda::make_device_buffer<int>(stream, device, cuda::std::array{11, -9, 27}));

  cuda::mr::legacy_managed_memory_resource managed_resource;
  const cuda::std::array host_device_host_values{3, 14, -15, 92};
  const auto host_device_values = cuda::make_buffer<int>(stream, managed_resource, host_device_host_values);
  const auto empty_values       = cuda::make_device_buffer<int>(stream, device);

  const cuda::std::array initial_updated_host_values{1, 2, 3, 4};
  auto updated_values = cuda::make_device_buffer<int>(stream, device, initial_updated_host_values);

  stream.sync();
  inspect_normal(normal_values);
  inspect_alias(aliased_values);
  inspect_vector(buffer_vector);
  inspect_host_device(host_device_values);
  inspect_empty(empty_values);
  inspect_before_update(updated_values);

  const cuda::std::array replacement_host_values{-8, 13, 21, -34};
  if (cudaMemcpyAsync(updated_values.data(),
                      replacement_host_values.data(),
                      replacement_host_values.size() * sizeof(*updated_values.data()),
                      cudaMemcpyDefault,
                      stream.get())
      != cudaSuccess)
  {
    return 1;
  }
  stream.sync();
  inspect_after_update(updated_values);
}
