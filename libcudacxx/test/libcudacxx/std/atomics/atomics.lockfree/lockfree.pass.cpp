//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// #define ATOMIC_BOOL_LOCK_FREE unspecified
// #define ATOMIC_CHAR_LOCK_FREE unspecified
// #define ATOMIC_CHAR16_T_LOCK_FREE unspecified
// #define ATOMIC_CHAR32_T_LOCK_FREE unspecified
// #define ATOMIC_WCHAR_T_LOCK_FREE unspecified
// #define ATOMIC_SHORT_LOCK_FREE unspecified
// #define ATOMIC_INT_LOCK_FREE unspecified
// #define ATOMIC_LONG_LOCK_FREE unspecified
// #define ATOMIC_LLONG_LOCK_FREE unspecified
// #define ATOMIC_POINTER_LOCK_FREE unspecified

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  assert(LIBCUDACXX_ATOMIC_BOOL_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_BOOL_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_BOOL_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_CHAR16_T_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_CHAR16_T_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_CHAR16_T_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_CHAR32_T_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_CHAR32_T_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_CHAR32_T_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_WCHAR_T_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_WCHAR_T_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_WCHAR_T_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_INT_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_INT_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_INT_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_LONG_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_LONG_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_LONG_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE == 2);

  return 0;
}
