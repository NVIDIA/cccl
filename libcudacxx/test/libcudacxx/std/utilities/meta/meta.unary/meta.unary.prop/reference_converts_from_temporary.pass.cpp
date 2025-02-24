//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_const

#include <cuda/std/type_traits>

#include "test_macros.h"

#ifdef _CCCL_BUILTIN_REFERENCE_CONVERTS_FROM_TEMPORARY

struct SimpleClass
{
  SimpleClass() = default;
};
struct ConvertsToRvalue
{
  constexpr operator int();
  explicit constexpr operator int&&();
};
struct ConvertsToConstReference
{
  constexpr operator int();
  explicit constexpr operator int&();
};

// not references
static_assert(cuda::std::reference_converts_from_temporary<int, int>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<void, void>::value == false, "");

// references do not bind
static_assert(cuda::std::reference_converts_from_temporary<int&, int>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&, int&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&, int&&>::value == false, "");

// references do not bind to non-convertible types
static_assert(cuda::std::reference_converts_from_temporary<int&, void>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&, const volatile void>::value == false, "");

// references do not bind to convertible types
static_assert(cuda::std::reference_converts_from_temporary<int&, long>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&, long&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&, long&&>::value == false, "");

// const references bind to values
static_assert(cuda::std::reference_converts_from_temporary<const int&, int>::value == true, "");

// const references do not bind to other references
static_assert(cuda::std::reference_converts_from_temporary<const int&, int&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<const int&, int&&>::value == false, "");

// const references bind to converted values
static_assert(cuda::std::reference_converts_from_temporary<const int&, long>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<const int&, long&>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<const int&, long&&>::value == true, "");

// rvalue references behave similar to const lvalue references
static_assert(cuda::std::reference_converts_from_temporary<int&&, int>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<int&&, int&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&&, int&&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&&, long>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<int&&, long&>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<int&&, long&&>::value == true, "");

// simple class types behave like builtin types
static_assert(cuda::std::reference_converts_from_temporary<SimpleClass&, SimpleClass>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<SimpleClass&, SimpleClass&&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<SimpleClass&, SimpleClass&&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<const SimpleClass&, SimpleClass>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<SimpleClass&&, SimpleClass>::value == true, "");

// No conversion possible
static_assert(cuda::std::reference_converts_from_temporary<const SimpleClass&, SimpleClass&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<const SimpleClass&, SimpleClass&&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<SimpleClass&&, SimpleClass&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<SimpleClass&&, SimpleClass&&>::value == false, "");

// arrays do not bind to references
static_assert(cuda::std::reference_converts_from_temporary<int&, int[]>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<const int&, int[]>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&&, int[]>::value == false, "");

// In contrast to reference_constructs_from_temporary conversions are possible
static_assert(cuda::std::reference_converts_from_temporary<int&&, ConvertsToRvalue>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<const int&, ConvertsToConstReference>::value == true, "");

#endif // _CCCL_BUILTIN_REFERENCE_CONVERTS_FROM_TEMPORARY

int main(int, char**)
{
  return 0;
}
