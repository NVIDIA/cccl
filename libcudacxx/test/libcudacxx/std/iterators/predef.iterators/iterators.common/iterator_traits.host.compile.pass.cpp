//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// The host `std::iterator_traits` of a `cuda::std::common_iterator` must be instantiable. With a host standard library
// that constrains `std::iterator_traits` on the iterator concepts (libstdc++ in C++20), instantiating it evaluates
// `cuda::std::iter_difference_t`, which must not query `std::iterator_traits` of the same iterator again. This is
// avoided by specializing `__is_primary_std_template` for `common_iterator`, which requires that trait to be a class
// template with every host standard library (it was an alias template with libc++).

#include <cuda/std/iterator>

#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
#  include <iterator>

using common_iter = cuda::std::common_iterator<int*, cuda::std::unreachable_sentinel_t>;

#  if !_CCCL_HOST_STD_LIB(LIBSTDCXX, <, 10)
static_assert(sizeof(std::iterator_traits<common_iter>) > 0);
#  endif // !_CCCL_HOST_STD_LIB(LIBSTDCXX, <, 10)
#endif // !TEST_COMPILER(NVRTC)

int main(int, char**)
{
  return 0;
}
