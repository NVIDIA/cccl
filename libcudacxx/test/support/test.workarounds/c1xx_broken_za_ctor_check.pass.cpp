//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verify TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK.

#include <cuda/std/type_traits>

#include "test_macros.h"
#include "test_workarounds.h"

struct X
{
  TEST_FUNC X(int) {}

  X(X&&)            = default;
  X& operator=(X&&) = default;

private:
  X(const X&)            = default;
  X& operator=(const X&) = default;
};

TEST_FUNC void PushFront(X&&) {}

template <class T = int>
TEST_FUNC auto test(int) -> decltype(PushFront(cuda::std::declval<T>()), cuda::std::true_type{});
TEST_FUNC auto test(long) -> cuda::std::false_type;

int main(int, char**)
{
#if defined(TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK)
  static_assert(!decltype(test(0))::value);
#else
  static_assert(decltype(test(0))::value);
#endif

  return 0;
}
