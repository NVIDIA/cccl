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

#include "test_macros.h"
#include <nv/target>

template <class T>
__host__ __device__ void
test_aligned_alloc(bool expect_success, cuda::std::size_t n, cuda::std::size_t align = TEST_ALIGNOF(T))
{
  if (expect_success)
  {
    static_assert(noexcept(cuda::std::aligned_alloc(n * sizeof(T), align)), "");

    T* ptr = static_cast<T*>(cuda::std::aligned_alloc(n * sizeof(T), align));

    // check that the memory was allocated
    assert(ptr != nullptr);

    // check memory alignment
    assert(((align - 1) & reinterpret_cast<cuda::std::uintptr_t>(ptr)) == 0);

    cuda::std::free(ptr);
  }
  else
  {
    T* ptr = static_cast<T*>(cuda::std::aligned_alloc(n * sizeof(T), align));

    // check that the memory allocation failed
    assert(ptr == nullptr);
  }
}

struct BigStruct
{
  int data[32];
};

struct TEST_ALIGNAS(cuda::std::max_align_t) AlignedStruct
{
  char data[32];
};

struct TEST_ALIGNAS(128) OverAlignedStruct
{
  char data[32];
};

__host__ __device__ bool should_expect_success()
{
  bool host_has_aligned_alloc = false;
#if TEST_STD_VER >= 2017 && !_CCCL_COMPILER(MSVC)
  host_has_aligned_alloc = true;
#endif // ^^^ TEST_STD_VER >= 2017 && !_CCCL_COMPILER(MSVC) ^^^

  bool device_has_aligned_alloc = false;
#if !_CCCL_CUDA_COMPILER(CLANG)
  device_has_aligned_alloc = true;
#endif // ^^^ !_CCCL_CUDA_COMPILER(CLANG) ^^^

  unused(host_has_aligned_alloc, device_has_aligned_alloc);

  NV_IF_ELSE_TARGET(NV_IS_HOST, (return host_has_aligned_alloc;), (return device_has_aligned_alloc;))
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
