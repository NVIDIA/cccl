//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// struct nothrow_t {
//   explicit nothrow_t() = default;
// };
// extern const nothrow_t nothrow;

// This test checks for https://wg21.link/LWG2510.

#include <cuda/std/__new>

cuda::std::nothrow_t f() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}

int main(int, char**) {
    return 0;
}
