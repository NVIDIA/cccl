//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// struct nothrow_t {
//   explicit nothrow_t() = default;
// };
// extern const nothrow_t nothrow;

#include <cuda/std/__new>

#include "test_macros.h"


int main(int, char**) {
  cuda::std::nothrow_t x = cuda::std::nothrow;
  unused(x);

  return 0;
}
