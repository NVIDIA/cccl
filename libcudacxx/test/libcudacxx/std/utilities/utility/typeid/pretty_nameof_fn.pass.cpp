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

// template <auto _Fn>
// cuda::std::__string_view cuda::std::__pretty_nameof_fn() noexcept

#include <cuda/std/__utility/typeid.h>
#include <cuda/std/cassert>

#include "test_macros.h"

// Functions referenced only by taking their address; never defined or called.
TEST_FUNC int a_free_function(double);
TEST_FUNC void another_free_function();

namespace a_namespace
{
TEST_FUNC int a_function_in_a_namespace(int);
} // namespace a_namespace

// The smoke test in the header is guarded the same way; mirror that here.
#if !defined(_CCCL_NO_CONSTEXPR_PRETTY_NAMEOF) && !defined(_CCCL_BROKEN_MSVC_FUNCSIG)

static_assert(cuda::std::__pretty_nameof_fn<a_free_function>() == cuda::std::__string_view("a_free_function"), "");

// The function can also be passed as a pointer; the result is identical.
static_assert(cuda::std::__pretty_nameof_fn<&a_free_function>() == cuda::std::__string_view("a_free_function"), "");

static_assert(
  cuda::std::__pretty_nameof_fn<another_free_function>() == cuda::std::__string_view("another_free_function"), "");

// Namespace qualifiers are kept, matching __pretty_nameof for types.
static_assert(cuda::std::__pretty_nameof_fn<a_namespace::a_function_in_a_namespace>()
                == cuda::std::__string_view("a_namespace::a_function_in_a_namespace"),
              "");

// Distinct functions yield distinct names.
static_assert(
  cuda::std::__pretty_nameof_fn<a_free_function>() != cuda::std::__pretty_nameof_fn<another_free_function>(), "");

#endif // !_CCCL_NO_CONSTEXPR_PRETTY_NAMEOF && !_CCCL_BROKEN_MSVC_FUNCSIG

int main(int, char**)
{
#if !defined(_CCCL_BROKEN_MSVC_FUNCSIG)
  // Runtime checks on the host. The constexpr path is exercised by the
  // static_asserts above (in both the host and device compilation passes).
  NV_IF_TARGET(
    NV_IS_HOST,
    (assert(cuda::std::__pretty_nameof_fn<a_free_function>() == cuda::std::__string_view("a_free_function"));
     assert(cuda::std::__pretty_nameof_fn<&a_free_function>() == cuda::std::__string_view("a_free_function"));
     assert(cuda::std::__pretty_nameof_fn<a_namespace::a_function_in_a_namespace>()
            == cuda::std::__string_view("a_namespace::a_function_in_a_namespace"));))
#endif // !_CCCL_BROKEN_MSVC_FUNCSIG

  return 0;
}
