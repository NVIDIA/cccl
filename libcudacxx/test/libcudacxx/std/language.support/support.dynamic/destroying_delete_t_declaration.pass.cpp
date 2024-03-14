//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// struct destroying_delete_t {
//   explicit destroying_delete_t() = default;
// };
// inline constexpr destroying_delete_t destroying_delete{};

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Test only the library parts of destroying delete in this test.
// Verify that it's properly declared after C++17 and that it's constexpr.
//
// Other tests will check the language side of things -- but those are
// limited to newer compilers.

#include <cuda/std/__new>

#include <cuda/std/cassert>
#include "test_macros.h"
#include "test_convertible.h"

#if defined(__cpp_impl_destroying_delete) &&                                   \
    defined(__cpp_lib_destroying_delete)
#ifndef __cccl_lib_destroying_delete
#error "Expected __cccl_lib_destroying_delete to be defined"
#elif __cccl_lib_destroying_delete < 201806L
#error "Unexpected value of __cccl_lib_destroying_delete"
#endif
#else
#ifdef __cccl_lib_destroying_delete
#error                                                                         \
    "__cccl_lib_destroying_delete should not be defined unless the compiler supports it"
#endif
#endif

#if defined(__cpp_impl_destroying_delete) &&                                   \
    defined(__cpp_lib_destroying_delete)

constexpr bool test_constexpr(cuda::std::destroying_delete_t) { return true; }

static_assert(
    cuda::std::is_default_constructible<cuda::std::destroying_delete_t>::value,
    "");
static_assert(!test_convertible<cuda::std::destroying_delete_t>(), "");
constexpr cuda::std::destroying_delete_t dd{};
static_assert(&dd != &cuda::std::destroying_delete, "");
static_assert(test_constexpr(cuda::std::destroying_delete), "");

#endif // __cpp_impl_destroying_delete && __cpp_lib_destroying_delete

int main(int, char**) { return 0; }
