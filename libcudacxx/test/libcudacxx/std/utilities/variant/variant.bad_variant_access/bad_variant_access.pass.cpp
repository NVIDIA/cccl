//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8
// UNSUPPORTED: true

// <cuda/std/variant>

/*

 class bad_variant_access : public exception {
public:
  bad_variant_access() noexcept;
  virtual const char* what() const noexcept;
};

*/

#include <cuda/std/cassert>
#include <cuda/std/exception>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::is_base_of<cuda::std::exception, cuda::std::bad_variant_access>::value, "");
  static_assert(noexcept(cuda::std::bad_variant_access{}), "must be noexcept");
  static_assert(noexcept(cuda::std::bad_variant_access{}.what()), "must be noexcept");
  cuda::std::bad_variant_access ex;
  assert(ex.what());

  return 0;
}
