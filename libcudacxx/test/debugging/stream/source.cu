// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda_runtime_api.h>

template <class T>
[[gnu::noinline]] void keep_for_debugger(const T& value)
{
  asm volatile("" : : "g"(&value) : "memory");
}

[[gnu::noinline]] void inspect_owning(const cuda::stream& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_ref(const cuda::stream_ref& value)
{
  keep_for_debugger(value);
}

using stream_ref_alias = cuda::stream_ref;

[[gnu::noinline]] void inspect_alias(const stream_ref_alias& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_default(const cuda::stream_ref& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_legacy(const cuda::stream_ref& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_per_thread(const cuda::stream_ref& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_invalid(const cuda::stream_ref& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_moved_from(const cuda::stream& value)
{
  keep_for_debugger(value);
}

int main()
{
  constexpr cuda::device_ref device{0};
  const cuda::stream owning_stream{device};
  const cuda::stream_ref stream_reference{owning_stream};
  const stream_ref_alias aliased_stream{owning_stream};
  const cuda::stream_ref default_stream{cudaStream_t{}};
  const cuda::stream_ref legacy_stream{cudaStreamLegacy};
  const cuda::stream_ref per_thread_stream{cudaStreamPerThread};
  const cuda::stream_ref invalid_stream{cuda::invalid_stream};
  cuda::stream moved_from_stream{device};
  const cuda::stream moved_to_stream{cuda::std::move(moved_from_stream)};

  inspect_owning(owning_stream);
  inspect_ref(stream_reference);
  inspect_alias(aliased_stream);
  inspect_default(default_stream);
  inspect_legacy(legacy_stream);
  inspect_per_thread(per_thread_stream);
  inspect_invalid(invalid_stream);
  inspect_moved_from(moved_from_stream);
  keep_for_debugger(moved_to_stream);
}
