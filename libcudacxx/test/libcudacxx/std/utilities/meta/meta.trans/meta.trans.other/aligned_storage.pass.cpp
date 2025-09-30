//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuda::std::pod is deprecated in C++20
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_IGNORE_DEPRECATED_API

// type_traits

// aligned_storage
//
//  Issue 3034 added:
//  The member typedef type shall be a trivial standard-layout type.

#include <cuda/std/cstddef> // for cuda::std::max_align_t
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  {
    typedef cuda::std::aligned_storage<10, 1>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<10, 1>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 1, "");
    static_assert(sizeof(T1) == 10, "");
  }
  {
    typedef cuda::std::aligned_storage<10, 2>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<10, 2>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 10, "");
  }
  {
    typedef cuda::std::aligned_storage<10, 4>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<10, 4>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 12, "");
  }
  {
    typedef cuda::std::aligned_storage<10, 8>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<10, 8>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
  }
  {
    typedef cuda::std::aligned_storage<10, 16>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<10, 16>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 16, "");
    static_assert(sizeof(T1) == 16, "");
  }
  {
    typedef cuda::std::aligned_storage<10, 32>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<10, 32>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 32, "");
  }
  {
    typedef cuda::std::aligned_storage<20, 32>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<20, 32>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 32, "");
  }
  {
    typedef cuda::std::aligned_storage<40, 32>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<40, 32>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 32, "");
    static_assert(sizeof(T1) == 64, "");
  }
  {
    typedef cuda::std::aligned_storage<12, 16>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<12, 16>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 16, "");
    static_assert(sizeof(T1) == 16, "");
  }
  {
    typedef cuda::std::aligned_storage<1>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<1>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 1, "");
    static_assert(sizeof(T1) == 1, "");
  }
  {
    typedef cuda::std::aligned_storage<2>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<2>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 2, "");
  }
  {
    typedef cuda::std::aligned_storage<3>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<3>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 2, "");
    static_assert(sizeof(T1) == 4, "");
  }
  {
    typedef cuda::std::aligned_storage<4>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<4>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 4, "");
  }
  {
    typedef cuda::std::aligned_storage<5>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<5>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 8, "");
  }
  {
    typedef cuda::std::aligned_storage<7>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<7>>);
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 4, "");
    static_assert(sizeof(T1) == 8, "");
  }
  {
    typedef cuda::std::aligned_storage<8>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<8>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 8, "");
  }
  {
    typedef cuda::std::aligned_storage<9>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<9>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
  }
  {
    typedef cuda::std::aligned_storage<15>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<15>>);
    static_assert(cuda::std::is_pod<T1>::value, "");
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
  }
  // Use alignof(cuda::std::max_align_t) below to find the max alignment instead of
  // hardcoding it, because it's different on different platforms.
  // (For example 8 on arm and 16 on x86.)
  {
    typedef cuda::std::aligned_storage<16>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<16>>);
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == alignof(cuda::std::max_align_t), "");
    static_assert(sizeof(T1) == 16, "");
  }
  {
    typedef cuda::std::aligned_storage<17>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<17>>);
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == alignof(cuda::std::max_align_t), "");
    static_assert(sizeof(T1) == 16 + alignof(cuda::std::max_align_t), "");
  }
  {
    typedef cuda::std::aligned_storage<10>::type T1;
    static_assert(cuda::std::is_same_v<T1, cuda::std::aligned_storage_t<10>>);
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == 8, "");
    static_assert(sizeof(T1) == 16, "");
  }
// NVCC doesn't support types that are _this_ overaligned, it seems
#if !TEST_CUDA_COMPILER(NVCC) && !TEST_COMPILER(NVRTC)
  {
    const int Align = 65536;
    typedef typename cuda::std::aligned_storage<1, Align>::type T1;
    static_assert(cuda::std::is_trivial<T1>::value, "");
    static_assert(cuda::std::is_standard_layout<T1>::value, "");
    static_assert(cuda::std::alignment_of<T1>::value == Align, "");
    static_assert(sizeof(T1) == Align, "");
  }
#endif // !TEST_CUDA_COMPILER(NVCC) && !TEST_COMPILER(NVRTC)

  return 0;
}
