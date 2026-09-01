// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/array>
#include <cuda/stream>

#include <vector>

#include <cuda_runtime_api.h>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

[[gnu::noinline]] void inspect_normal(const cuda::device_buffer<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

using device_buffer_alias = cuda::buffer<int, cuda::mr::device_accessible>;

[[gnu::noinline]] void inspect_alias(const device_buffer_alias& values)
{
  KEEP_FOR_DEBUGGER(values);
}

// An optimized build inlines every use of std::vector::operator[], so no
// out-of-line copy is left for the debugger to call. Instantiate that one member
// explicitly to keep a copy, which lets the debugger evaluate values[i]. A full
// class instantiation does not compile because the buffer is move-only.
template typename std::vector<cuda::device_buffer<int>>::const_reference
  std::vector<cuda::device_buffer<int>>::operator[](size_type) const;

[[gnu::noinline]] void inspect_vector(const std::vector<cuda::device_buffer<int>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

template <class Buffer>
[[gnu::noinline]] void inspect_host_device(const Buffer& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_empty(const cuda::device_buffer<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::device_buffer<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_after_update(const cuda::device_buffer<int>& values)
{
  KEEP_FOR_DEBUGGER(values);
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
