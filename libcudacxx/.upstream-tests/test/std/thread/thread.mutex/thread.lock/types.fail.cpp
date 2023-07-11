//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-threads
// UNSUPPORTED: nvrtc
// UNSUPPORTED: pre-sm-70

// <mutex>

// struct defer_lock_t { explicit defer_lock_t() = default; };
// struct try_to_lock_t { explicit try_to_lock_t() = default; };
// struct adopt_lock_t { explicit adopt_lock_t() = default; };

// This test checks for LWG 2510.

#include<cuda/std/mutex>


cuda::std::defer_lock_t __host__ __device__ f1() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
cuda::std::try_to_lock_t __host__ __device__ f2() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
cuda::std::adopt_lock_t __host__ __device__ f3() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}

int main(int, char**) {
    return 0;
}
