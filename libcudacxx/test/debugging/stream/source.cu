// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/utility>
#include <cuda/stream>

#include <vector>

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

[[gnu::noinline]] void inspect_summary(const std::vector<cuda::stream_ref>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::stream_ref& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_after_update(const cuda::stream_ref& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_capturing(const cuda::stream& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_after_capture(const cuda::stream& value)
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
  const std::vector<cuda::stream_ref> summarized_streams{stream_reference, default_stream, invalid_stream};
  cuda::stream_ref updated_stream{default_stream};
  const cuda::stream capture_stream{device};

  inspect_owning(owning_stream);
  inspect_ref(stream_reference);
  inspect_alias(aliased_stream);
  inspect_default(default_stream);
  inspect_legacy(legacy_stream);
  inspect_per_thread(per_thread_stream);
  inspect_invalid(invalid_stream);
  inspect_moved_from(moved_from_stream);
  inspect_summary(summarized_streams);
  inspect_before_update(updated_stream);
  updated_stream = owning_stream;
  inspect_after_update(updated_stream);

  if (cudaStreamBeginCapture(capture_stream.get(), cudaStreamCaptureModeGlobal) != cudaSuccess)
  {
    return 1;
  }

  // Printing at this breakpoint exercises the pretty-printers during global
  // capture. A successful cudaStreamEndCapture below verifies that debugger
  // inferior calls did not invalidate the capture.
  inspect_capturing(capture_stream);
  cudaGraph_t graph{};
  const cudaError_t end_capture_status = cudaStreamEndCapture(capture_stream.get(), &graph);
  if (end_capture_status != cudaSuccess)
  {
    return 1;
  }
  if (cudaGraphDestroy(graph) != cudaSuccess)
  {
    return 1;
  }
  inspect_after_capture(capture_stream);

  keep_for_debugger(moved_to_stream);
  return 0;
}
