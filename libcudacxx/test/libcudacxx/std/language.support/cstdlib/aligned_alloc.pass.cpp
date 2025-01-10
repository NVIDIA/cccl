//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/cstdlib>
#include <cuda/std/limits>

#include "test_macros.h"

template <class T>
_LIBCUDACXX_ALIGNED_ALLOC_EXSPACE void
test_aligned_alloc_success(cuda::std::size_t n, cuda::std::size_t align = TEST_ALIGNOF(T))
{
#if (TEST_STD_VER >= 17 && !_CCCL_COMPILER(MSVC)) || (_CCCL_HAS_CUDA_COMPILER && !_CCCL_CUDA_COMPILER(CLANG))
  static_assert(noexcept(cuda::std::aligned_alloc(n * sizeof(T), align)), "");

  T* ptr = static_cast<T*>(cuda::std::aligned_alloc(n * sizeof(T), align));

  // check that the memory was allocated
  assert(ptr != nullptr);

  // check memory alignment
  assert(((align - 1) & reinterpret_cast<cuda::std::uintptr_t>(ptr)) == 0);

  cuda::std::free(ptr);
#endif // (TEST_STD_VER >= 17 && !_CCCL_COMPILER(MSVC)) || (_CCCL_HAS_CUDA_COMPILER && !_CCCL_CUDA_COMPILER(CLANG))
}

template <class T>
_LIBCUDACXX_ALIGNED_ALLOC_EXSPACE void
test_aligned_alloc_fail(cuda::std::size_t n, cuda::std::size_t align = TEST_ALIGNOF(T))
{
#if (TEST_STD_VER >= 17 && !_CCCL_COMPILER(MSVC)) || (_CCCL_HAS_CUDA_COMPILER && !_CCCL_CUDA_COMPILER(CLANG))
  T* ptr = static_cast<T*>(cuda::std::aligned_alloc(n * sizeof(T), align));

  // check that the memory allocation failed
  assert(ptr == nullptr);
#endif // (TEST_STD_VER >= 17 && !_CCCL_COMPILER(MSVC)) || (_CCCL_HAS_CUDA_COMPILER && !_CCCL_CUDA_COMPILER(C
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

_LIBCUDACXX_ALIGNED_ALLOC_EXSPACE void test()
{
  test_aligned_alloc_success<int>(10, 4);
  test_aligned_alloc_success<char>(128, 8);
  test_aligned_alloc_success<double>(8, 32);
  test_aligned_alloc_success<BigStruct>(4, 128);
  test_aligned_alloc_success<AlignedStruct>(16);
  test_aligned_alloc_success<OverAlignedStruct>(1);
  test_aligned_alloc_success<OverAlignedStruct>(1, 256);

  test_aligned_alloc_fail<int>(10, 3);
}

int main(int, char**)
{
#if _LIBCUDACXX_HAS_ALIGNED_ALLOC_HOST
  NV_IF_TARGET(NV_IS_HOST, test();)
#endif // _LIBCUDACXX_HAS_ALIGNED_ALLOC_HOST
#if _LIBCUDACXX_HAS_ALIGNED_ALLOC_DEVICE
  NV_IF_TARGET(NV_IS_DEVICE, test();)
#endif // _LIBCUDACXX_HAS_ALIGNED_ALLOC_DEVICE

  return 0;
}
