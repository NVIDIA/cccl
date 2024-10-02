//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <cuda/std/type_traits>

// __libcpp_is_constant_evaluated()

// returns false when there's no constant evaluation support from the compiler.
//  as well as when called not in a constexpr context

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  ASSERT_SAME_TYPE(decltype(cuda::std::__libcpp_is_constant_evaluated()), bool);
  ASSERT_NOEXCEPT(cuda::std::__libcpp_is_constant_evaluated());

#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(cuda::std::__libcpp_is_constant_evaluated(), "");
#endif

  bool p = cuda::std::__libcpp_is_constant_evaluated();
  assert(!p);

  return 0;
}
