//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// enum class align_val_t : size_t {}

// UNSUPPORTED: c++03, c++11, c++14

// Libc++ when built for z/OS doesn't contain the aligned allocation functions,
// nor does the dynamic library shipped with z/OS.
// UNSUPPORTED: target={{.+}}-zos{{.*}}

#include <cuda/std/__new>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**) {
  {
    static_assert(cuda::std::is_enum<cuda::std::align_val_t>::value, "");
    static_assert(cuda::std::is_same<cuda::std::underlying_type<cuda::std::align_val_t>::type, cuda::std::size_t>::value, "");
    static_assert(!cuda::std::is_constructible<cuda::std::align_val_t, cuda::std::size_t>::value, "");
    static_assert(!cuda::std::is_constructible<cuda::std::size_t, cuda::std::align_val_t>::value, "");
  }
  {
    constexpr auto a = cuda::std::align_val_t(0);
    constexpr auto b = cuda::std::align_val_t(32);
    constexpr auto c = cuda::std::align_val_t(-1);
    static_assert(a != b, "");
    static_assert(a == cuda::std::align_val_t(0), "");
    static_assert(b == cuda::std::align_val_t(32), "");
    static_assert(static_cast<cuda::std::size_t>(c) == (cuda::std::size_t)-1, "");
  }

  return 0;
}
