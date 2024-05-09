//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/ctime>
#include <cuda/std/type_traits>

#include "test_macros.h"

#ifndef NULL
#  error NULL not defined
#endif

#ifndef TEST_COMPILER_NVRTC
#  ifndef CLOCKS_PER_SEC
#    error CLOCKS_PER_SEC not defined
#  endif
#endif

#if TEST_STD_VER > 2014 && defined(TEST_HAS_C11_FEATURES)
#  ifndef TIME_UTC
#    error TIME_UTC not defined
#  endif
#endif

#if defined(__GNUC__)
#  pragma GCC diagnostic ignored "-Wformat-zero-length"
#endif

TEST_NV_DIAG_SUPPRESS(set_but_not_used)

int main(int, char**)
{
  cuda::std::clock_t c = 0;
  cuda::std::size_t s  = 0;
  cuda::std::time_t t  = 0;
  unused(c); // Prevent unused warning
  unused(s); // Prevent unused warning
  unused(t); // Prevent unused warning
#ifndef TEST_COMPILER_NVRTC
  cuda::std::tm tm = {};
  char str[3];
  unused(tm); // Prevent unused warning
  unused(str); // Prevent unused warning
#  if TEST_STD_VER > 2014 && defined(TEST_HAS_C11_FEATURES)
  cuda::std::timespec tmspec = {};
  unused(tmspec); // Prevent unused warning
#  endif

#  ifndef TEST_COMPILER_CLANG_CUDA
  static_assert((cuda::std::is_same<decltype(cuda::std::clock()), cuda::std::clock_t>::value), "");
#  endif // TEST_COMPILER_CLANG_CUDA
  static_assert((cuda::std::is_same<decltype(cuda::std::difftime(t, t)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::mktime(&tm)), cuda::std::time_t>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::time(&t)), cuda::std::time_t>::value), "");
#  if TEST_STD_VER > 2014 && defined(TEST_HAS_TIMESPEC_GET)
  static_assert((cuda::std::is_same<decltype(cuda::std::timespec_get(&tmspec, 0)), int>::value), "");
#  endif
#  ifndef _LIBCUDACXX_HAS_NO_THREAD_UNSAFE_C_FUNCTIONS
  static_assert((cuda::std::is_same<decltype(cuda::std::asctime(&tm)), char*>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::ctime(&t)), char*>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::gmtime(&t)), cuda::std::tm*>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::localtime(&t)), cuda::std::tm*>::value), "");
#  endif
  static_assert((cuda::std::is_same<decltype(cuda::std::strftime(str, s, "", &tm)), cuda::std::size_t>::value), "");
#endif

  return 0;
}
