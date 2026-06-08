//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile
// error: taking address of a function is unsupported in tile code

// <cuda/std/__utility/typeid.h>

// template <auto _Vp>
// cuda::std::__string_view cuda::std::__stringize() noexcept

#include <cuda/std/__utility/typeid.h>
#include <cuda/std/cassert>

#include "test_macros.h"

// Functions referenced only by taking their address; never defined or called.
TEST_FUNC int a_free_function(double);

namespace a_namespace
{
TEST_FUNC int a_function_in_a_namespace(int);
} // namespace a_namespace

#if !defined(__CUDACC__)
// An overloaded function must be disambiguated with a cast to select one
// overload. This is only exercised by a plain host compiler: under nvcc,
// cudafe++ rewrites the cast back to the (ambiguous) `&an_overloaded_function`,
// so a cast-disambiguated overload cannot be used as a template argument there.
int an_overloaded_function(int);
int an_overloaded_function(double);
#endif // !__CUDACC__

// The exact spelling of a value is not guaranteed to be identical across
// compilers (e.g. an unsigned literal might be spelled "42" or "42U", a char
// "'A'" or 65, an enumerator by name or by a cast, and a bool "true" or "1").
// Integer literals are spelled identically everywhere, so those are the values
// checked exactly here; everything else uses distinctness or substring checks.

// The smoke test in the header is guarded the same way; mirror that here.
#if !defined(_CCCL_NO_CONSTEXPR_PRETTY_NAMEOF) && !defined(_CCCL_BROKEN_MSVC_FUNCSIG)

// Integer literals: spelled identically on every supported compiler.
static_assert(cuda::std::__stringize<42>() == cuda::std::__string_view("42"), "");
static_assert(cuda::std::__stringize<-7>() == cuda::std::__string_view("-7"), "");
static_assert(cuda::std::__stringize<0>() == cuda::std::__string_view("0"), "");

// Other values: only require distinct values to have distinct spellings.
static_assert(cuda::std::__stringize<42>() != cuda::std::__stringize<43>(), "");
static_assert(cuda::std::__stringize<true>() != cuda::std::__stringize<false>(), "");

// Functions: the (possibly qualified) name appears, with the leading '&' dropped.
static_assert(cuda::std::__stringize<a_free_function>().find("a_free_function") != -1, "");
static_assert(cuda::std::__stringize<&a_free_function>().find("a_free_function") != -1, ""); // pointer form
static_assert(cuda::std::__stringize<a_namespace::a_function_in_a_namespace>().find("a_function_in_a_namespace") != -1,
              "");
static_assert(
  cuda::std::__stringize<a_free_function>() != cuda::std::__stringize<a_namespace::a_function_in_a_namespace>(), "");

#  if !TEST_COMPILER(MSVC)
// On compilers with a stable spelling, check the exact function name: the leading
// '&' is dropped and namespace qualifiers are kept.
static_assert(cuda::std::__stringize<a_free_function>() == cuda::std::__string_view("a_free_function"), "");
static_assert(cuda::std::__stringize<&a_free_function>() == cuda::std::__string_view("a_free_function"), "");
static_assert(cuda::std::__stringize<a_namespace::a_function_in_a_namespace>()
                == cuda::std::__string_view("a_namespace::a_function_in_a_namespace"),
              "");
#  endif // !TEST_COMPILER(MSVC)

#  if !defined(__CUDACC__)
// An overloaded function: a cast picks the overload, and every overload spells
// out the same (the function's) name. See the note on an_overloaded_function.
static_assert(cuda::std::__stringize<static_cast<int (*)(int)>(an_overloaded_function)>()
                == cuda::std::__string_view("an_overloaded_function"),
              "");
static_assert(cuda::std::__stringize<static_cast<int (*)(double)>(an_overloaded_function)>()
                == cuda::std::__string_view("an_overloaded_function"),
              "");
#  endif // !__CUDACC__

#endif // !_CCCL_NO_CONSTEXPR_PRETTY_NAMEOF && !_CCCL_BROKEN_MSVC_FUNCSIG

int main(int, char**)
{
#if !defined(_CCCL_BROKEN_MSVC_FUNCSIG)
  // Runtime checks on the host. The constexpr path is exercised by the
  // static_asserts above (in both the host and device compilation passes).
  NV_IF_TARGET(
    NV_IS_HOST,
    (assert(cuda::std::__stringize<42>() == cuda::std::__string_view("42"));
     assert(cuda::std::__stringize<-7>() == cuda::std::__string_view("-7"));
     assert(cuda::std::__stringize<42>() != cuda::std::__stringize<43>());
     assert(cuda::std::__stringize<true>() != cuda::std::__stringize<false>());
     assert(cuda::std::__stringize<a_free_function>().find("a_free_function") != -1);
     assert(cuda::std::__stringize<&a_free_function>().find("a_free_function") != -1);
     assert(cuda::std::__stringize<a_namespace::a_function_in_a_namespace>().find("a_function_in_a_namespace") != -1);))
#endif // !_CCCL_BROKEN_MSVC_FUNCSIG

  return 0;
}
