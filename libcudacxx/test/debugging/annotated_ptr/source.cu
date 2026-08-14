// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/annotated_ptr>

template <class T>
[[gnu::noinline]] void keep_for_debugger(const T& value)
{
  asm volatile("" : : "g"(&value) : "memory");
}

// Type aliases for annotated_ptr instances
using streaming_annotated_int  = cuda::annotated_ptr<int, cuda::access_property::streaming>;
using persisting_annotated_int = cuda::annotated_ptr<int, cuda::access_property::persisting>;
using global_annotated_float   = cuda::annotated_ptr<float, cuda::access_property::global>;

// Test a streaming annotated_ptr
[[gnu::noinline]] void inspect_streaming(const streaming_annotated_int& ptr)
{
  keep_for_debugger(ptr);
}

// Test a persisting annotated_ptr
[[gnu::noinline]] void inspect_persisting(const persisting_annotated_int& ptr)
{
  keep_for_debugger(ptr);
}

// Test a global annotated_ptr
[[gnu::noinline]] void inspect_global(const global_annotated_float& ptr)
{
  keep_for_debugger(ptr);
}

// Test a null streaming_annotated_ptr
[[gnu::noinline]] void inspect_null(const streaming_annotated_int& ptr)
{
  keep_for_debugger(ptr);
}

int main()
{
  // Allocate test data with mutable backing (required by annotated_ptr)
  int streaming_data  = 42;
  int persisting_data = 100;
  float global_data   = 3.14f;

  // Test case 1: streaming annotated_ptr
  const streaming_annotated_int streaming_ptr(&streaming_data);
  inspect_streaming(streaming_ptr);

  // Test case 2: persisting annotated_ptr
  const persisting_annotated_int persisting_ptr(&persisting_data);
  inspect_persisting(persisting_ptr);

  // Test case 3: global annotated_ptr
  const global_annotated_float global_ptr(&global_data);
  inspect_global(global_ptr);

  // Test case 4: null streaming_annotated_ptr (use default constructor)
  const streaming_annotated_int null_ptr{};
  inspect_null(null_ptr);

  return 0;
}
