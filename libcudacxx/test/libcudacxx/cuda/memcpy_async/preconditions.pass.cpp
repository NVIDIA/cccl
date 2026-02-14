//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#define _LIBCUDACXX_MEMCPY_ASYNC_PRE_TESTING

#include <cuda/__memcpy_async/check_preconditions.h>
#include <cuda/std/cstddef>

__host__ __device__ void test()
{
  using T = int;

  constexpr cuda::std::size_t align_scale = 2;
  constexpr cuda::std::size_t align       = align_scale * alignof(T);
  constexpr cuda::std::size_t n           = 16;
  constexpr cuda::std::size_t size        = n * sizeof(T);

  // test typed overloads
  {
    alignas(align) T a[n * 2]{};
    alignas(align) const T b[n * 2]{};

    const auto a_missaligned = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(a) + alignof(T) / 2);
    const auto b_missaligned = reinterpret_cast<const T*>(reinterpret_cast<uintptr_t>(b) + alignof(T) / 2);

    // 1. test ordinary size type
    {
      assert(cuda::__memcpy_async_check_pre(a, b, size));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, b, size));
      assert(!cuda::__memcpy_async_check_pre(a, b_missaligned, size));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, b_missaligned, size));
    }

    // 2. test overaligned cuda::aligned_size_t
    {
      cuda::aligned_size_t<align> size_aligned(size);
      assert(cuda::__memcpy_async_check_pre(a, b, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, b, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a, b_missaligned, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, b_missaligned, size_aligned));
    }

    // 3. test cuda::aligned_size_t aligned to alignof(T)
    {
      cuda::aligned_size_t<align / align_scale> size_aligned(size);
      assert(cuda::__memcpy_async_check_pre(a, b, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, b, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a, b_missaligned, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, b_missaligned, size_aligned));
    }

    // 4. test underaligned cuda::aligned_size_t
    {
      cuda::aligned_size_t<align / (2 * align_scale)> size_aligned(size);
      assert(cuda::__memcpy_async_check_pre(a, b, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, b, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a, b_missaligned, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, b_missaligned, size_aligned));
    }

    // 5. test overlap
    {
      assert(!cuda::__memcpy_async_check_pre(a, a, size));
      assert(!cuda::__memcpy_async_check_pre(a, a_missaligned, size));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, a, size));
      assert(cuda::__memcpy_async_check_pre(a, a + n, size));
      assert(cuda::__memcpy_async_check_pre(a + n, a, size));
      assert(!cuda::__memcpy_async_check_pre(a, a + n - 1, size));
      assert(!cuda::__memcpy_async_check_pre(a + n - 1, a, size));
    }
  }

  // test void overloads
  {
    alignas(align) T a_buff[n * 2]{};
    alignas(align) const T b_buff[n * 2]{};

    void* a       = a_buff;
    const void* b = b_buff;

    const auto a_missaligned = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(a) + alignof(T) / 2);
    const auto b_missaligned = reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(b) + alignof(T) / 2);

    // 1. test ordinary size type
    {
      assert(cuda::__memcpy_async_check_pre(a, b, size));
      assert(cuda::__memcpy_async_check_pre(a_missaligned, b, size));
      assert(cuda::__memcpy_async_check_pre(a, b_missaligned, size));
      assert(cuda::__memcpy_async_check_pre(a_missaligned, b_missaligned, size));
    }

    // 2. test overaligned cuda::aligned_size_t
    {
      cuda::aligned_size_t<align> size_aligned(size);
      assert(cuda::__memcpy_async_check_pre(a, b, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, b, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a, b_missaligned, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, b_missaligned, size_aligned));
    }

    // 3. test cuda::aligned_size_t aligned to alignof(T)
    {
      cuda::aligned_size_t<align / align_scale> size_aligned(size);
      assert(cuda::__memcpy_async_check_pre(a, b, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, b, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a, b_missaligned, size_aligned));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, b_missaligned, size_aligned));
    }

    // 4. test underaligned cuda::aligned_size_t
    {
      cuda::aligned_size_t<align / (2 * align_scale)> size_aligned(size);
      assert(cuda::__memcpy_async_check_pre(a, b, size_aligned));
      assert(cuda::__memcpy_async_check_pre(a_missaligned, b, size_aligned));
      assert(cuda::__memcpy_async_check_pre(a, b_missaligned, size_aligned));
      assert(cuda::__memcpy_async_check_pre(a_missaligned, b_missaligned, size_aligned));
    }

    // 5. test overlap
    {
      assert(!cuda::__memcpy_async_check_pre(a, a, size));
      assert(!cuda::__memcpy_async_check_pre(a, a_missaligned, size));
      assert(!cuda::__memcpy_async_check_pre(a_missaligned, a, size));
      assert(cuda::__memcpy_async_check_pre(a, (const void*) (a_buff + n), size));
      assert(cuda::__memcpy_async_check_pre((void*) (a_buff + n), a, size));
      assert(!cuda::__memcpy_async_check_pre(a, (const void*) (a_buff + n - 1), size));
      assert(!cuda::__memcpy_async_check_pre((void*) (a_buff + n - 1), a, size));
    }
  }
}

int main(int, char**)
{
  test();
  return 0;
}
