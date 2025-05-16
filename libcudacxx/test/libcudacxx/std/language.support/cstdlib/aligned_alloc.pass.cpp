//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Disable CCCL assertions in this test to test the erroneous behavior
#undef CCCL_ENABLE_ASSERTIONS

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/cstdlib>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <nv/target>

#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4324) // padding was added at the end of a structure because of an alignment specifier

template <class T>
__host__ __device__ void
test_aligned_alloc(bool expect_success, cuda::std::size_t n, cuda::std::size_t align = alignof(T))
{
  static_assert(noexcept(cuda::std::aligned_alloc(n * sizeof(T), align)), "");
  T* ptr = static_cast<T*>(cuda::std::aligned_alloc(n * sizeof(T), align));
  if (expect_success)
  {
    // check that the memory was allocated
    assert(ptr != nullptr);

    // check memory alignment
    assert(((align - 1) & reinterpret_cast<cuda::std::uintptr_t>(ptr)) == 0);
  }
  else
  {
    // This is undefined behavior and dependent on the host libc
  }
  cuda::std::free(ptr);
}

struct BigStruct
{
  int data[32];
};

struct alignas(cuda::std::max_align_t) AlignedStruct
{
  char data[32];
};

struct alignas(128) OverAlignedStruct
{
  char data[32];
};

__host__ __device__ bool should_expect_success()
{
  bool host_expect_success = true;
#if TEST_COMPILER(MSVC)
  host_expect_success = false;
#endif // TEST_COMPILER(MSVC)

  unused(host_expect_success);

  NV_IF_ELSE_TARGET(NV_IS_HOST, (return host_expect_success;), (return true;))
}

__host__ __device__ void test()
{
  const bool expect_success = should_expect_success();

  test_aligned_alloc<int>(expect_success, 10, 4);
  test_aligned_alloc<char>(expect_success, 128, 8);
  test_aligned_alloc<double>(expect_success, 8, 32);
  test_aligned_alloc<BigStruct>(expect_success, 4, 128);
  test_aligned_alloc<AlignedStruct>(expect_success, 16);
  test_aligned_alloc<OverAlignedStruct>(expect_success, 1);
  test_aligned_alloc<OverAlignedStruct>(expect_success, 1, 256);

  test_aligned_alloc<int>(false, 10, 3);
}

int main(int, char**)
{
  test();
  return 0;
}
