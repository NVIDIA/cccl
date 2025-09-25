//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuda::std::pod is deprecated in C++20
// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

// max_align_t is a trivial standard-layout type whose alignment requirement
//   is at least as great as that of every scalar type
#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
#  include <stdio.h>
#endif // TEST_COMPILER(NVRTC)

int main(int, char**)
{
#if TEST_STD_VER > 2017
  //  P0767
  static_assert(cuda::std::is_trivial<cuda::std::max_align_t>::value,
                "cuda::std::is_trivial<cuda::std::max_align_t>::value");
  static_assert(cuda::std::is_standard_layout<cuda::std::max_align_t>::value,
                "cuda::std::is_standard_layout<cuda::std::max_align_t>::value");
#else
  static_assert(cuda::std::is_pod<cuda::std::max_align_t>::value, "cuda::std::is_pod<cuda::std::max_align_t>::value");
#endif
  static_assert((cuda::std::alignment_of<cuda::std::max_align_t>::value >= cuda::std::alignment_of<long long>::value),
                "cuda::std::alignment_of<cuda::std::max_align_t>::value >= "
                "cuda::std::alignment_of<long long>::value");
#if _CCCL_HAS_LONG_DOUBLE()
  static_assert(cuda::std::alignment_of<cuda::std::max_align_t>::value >= cuda::std::alignment_of<long double>::value,
                "cuda::std::alignment_of<cuda::std::max_align_t>::value >= "
                "cuda::std::alignment_of<long double>::value");
#endif // _CCCL_HAS_LONG_DOUBLE()
  static_assert(cuda::std::alignment_of<cuda::std::max_align_t>::value >= cuda::std::alignment_of<void*>::value,
                "cuda::std::alignment_of<cuda::std::max_align_t>::value >= "
                "cuda::std::alignment_of<void*>::value");

  return 0;
}
