// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/annotated_ptr>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

[[gnu::noinline]] void inspect_streaming(const cuda::annotated_ptr<int, cuda::access_property::streaming>& ptr)
{
  KEEP_FOR_DEBUGGER(ptr);
}

[[gnu::noinline]] void inspect_persisting(const cuda::annotated_ptr<int, cuda::access_property::persisting>& ptr)
{
  KEEP_FOR_DEBUGGER(ptr);
}

[[gnu::noinline]] void inspect_global(const cuda::annotated_ptr<float, cuda::access_property::global>& ptr)
{
  KEEP_FOR_DEBUGGER(ptr);
}

[[gnu::noinline]] void inspect_null(const cuda::annotated_ptr<int, cuda::access_property::streaming>& ptr)
{
  KEEP_FOR_DEBUGGER(ptr);
}

[[gnu::noinline]] void inspect_before_update(const cuda::annotated_ptr<int, cuda::access_property::streaming>& ptr)
{
  KEEP_FOR_DEBUGGER(ptr);
}

[[gnu::noinline]] void inspect_after_update(const cuda::annotated_ptr<int, cuda::access_property::streaming>& ptr)
{
  KEEP_FOR_DEBUGGER(ptr);
}

int main()
{
  // Allocate test data with mutable backing (required by annotated_ptr)
  int streaming_data  = 42;
  int persisting_data = 100;
  float global_data   = 3.14f;
  int updated_data    = 7;

  const cuda::annotated_ptr<int, cuda::access_property::streaming> streaming_ptr(&streaming_data);
  inspect_streaming(streaming_ptr);

  const cuda::annotated_ptr<int, cuda::access_property::persisting> persisting_ptr(&persisting_data);
  inspect_persisting(persisting_ptr);

  const cuda::annotated_ptr<float, cuda::access_property::global> global_ptr(&global_data);
  inspect_global(global_ptr);

  const cuda::annotated_ptr<int, cuda::access_property::streaming> null_ptr{};
  inspect_null(null_ptr);

  // The pointer is non-null before the update and null after it, so the two stops print
  // different values even though the runner masks non-zero addresses.
  cuda::annotated_ptr<int, cuda::access_property::streaming> updated_ptr(&updated_data);
  inspect_before_update(updated_ptr);
  updated_ptr = cuda::annotated_ptr<int, cuda::access_property::streaming>{};
  inspect_after_update(updated_ptr);

  return 0;
}
