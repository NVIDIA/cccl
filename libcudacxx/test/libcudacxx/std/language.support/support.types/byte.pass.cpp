//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include "test_macros.h"

// XFAIL: c++98, c++03, c++11

// If we're just building the test and not executing it, it should pass.
// UNSUPPORTED: no_execute

// cuda::std::byte is not an integer type, nor a character type.
// It is a distinct type for accessing the bits that ultimately make up object storage.

#if TEST_STD_VER > 2011
static_assert(cuda::std::is_trivial<cuda::std::byte>::value, ""); // P0767
#else
static_assert(cuda::std::is_pod<cuda::std::byte>::value, "");
#endif
static_assert(!cuda::std::is_arithmetic<cuda::std::byte>::value, "");
static_assert(!cuda::std::is_integral<cuda::std::byte>::value, "");

static_assert(!cuda::std::is_same<cuda::std::byte, char>::value, "");
static_assert(!cuda::std::is_same<cuda::std::byte, signed char>::value, "");
static_assert(!cuda::std::is_same<cuda::std::byte, unsigned char>::value, "");

// The standard doesn't outright say this, but it's pretty clear that it has to be true.
static_assert(sizeof(cuda::std::byte) == 1, "");

int main(int, char**)
{
  return 0;
}
