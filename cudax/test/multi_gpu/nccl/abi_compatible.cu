//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/underlying_type.h>

#include <cuda/experimental/__nccl/abi_compatible.h>

#include <c2h/catch2_test_helper.h>

namespace
{
namespace abi_detail = ::cuda::experimental::__nccl::__abi_detail;

// NOLINTBEGIN(bugprone-reserved-identifier)
enum Foo_enum
{
};

#if _CCCL_OS(WINDOWS)
using FooEnum = int;
#else
using FooEnum = unsigned int;
#endif

struct Foo_st;
using FooStruct = Foo_st*;

// An enum whose underlying type is fixed, so that mismatch tests are deterministic
// regardless of how the implementation picks the underlying type of an unfixed enum.
enum class CharEnum : char
{
};

enum class IntEnum : int
{
};

struct Bar_st;
// NOLINTEND(bugprone-reserved-identifier)
} // namespace

C2H_TEST("nccl __abi_compatible type comparisons", "[multi_gpu][nccl]")
{
  // --- Identical / scalar types -------------------------------------------------
  STATIC_REQUIRE(abi_detail::__abi_compatible<int, int>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<int, float>());

  // remove_cv only strips the top level, so a top-level cv difference is still compatible.
  STATIC_REQUIRE(abi_detail::__abi_compatible<const int, int>());
  STATIC_REQUIRE(abi_detail::__abi_compatible<volatile int, int>());
  STATIC_REQUIRE(abi_detail::__abi_compatible<const volatile int, int>());

  // Different-width / signedness integers are distinct types and must not be compatible.
  STATIC_REQUIRE(!abi_detail::__abi_compatible<int, long>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<int, short>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<int, unsigned int>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<char, signed char>());

  // --- Pointers -----------------------------------------------------------------
  STATIC_REQUIRE(abi_detail::__abi_compatible<const char**, const char* const*>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<const char*, const int*>());

  // Deep cv-qualifications on the pointee are stripped at each level of recursion.
  STATIC_REQUIRE(abi_detail::__abi_compatible<int*, const int*>());
  STATIC_REQUIRE(abi_detail::__abi_compatible<int**, const int* const*>());

  // Multi-level pointers must agree in depth.
  STATIC_REQUIRE(abi_detail::__abi_compatible<int**, int**>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<int**, int*>());

  // A pointer is never compatible with a non-pointer (the `&&` in the pointer branch).
  STATIC_REQUIRE(!abi_detail::__abi_compatible<int*, int>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<int, int*>());

  // void* is not compatible with a typed pointer: the pointees differ.
  STATIC_REQUIRE(!abi_detail::__abi_compatible<void*, int*>());
  STATIC_REQUIRE(abi_detail::__abi_compatible<void*, void*>());

  // Opaque struct pointers: identical handle types and matching opaque pointees.
  STATIC_REQUIRE(abi_detail::__abi_compatible<Foo_st*, FooStruct>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<Foo_st*, Bar_st*>());

  // --- Enums --------------------------------------------------------------------
  STATIC_REQUIRE(::cuda::std::is_same_v<FooEnum, ::cuda::std::underlying_type_t<Foo_enum>>);

  // enum vs its underlying type (either side), and enum-vs-enum through the pointee.
  STATIC_REQUIRE(abi_detail::__abi_compatible<FooEnum, Foo_enum>());
  STATIC_REQUIRE(abi_detail::__abi_compatible<Foo_enum, FooEnum>());
  STATIC_REQUIRE(abi_detail::__abi_compatible<FooEnum*, Foo_enum*>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<FooEnum*, Foo_enum>());

  // Both sides enums: compatible iff their underlying types match exactly.
  STATIC_REQUIRE(abi_detail::__abi_compatible<CharEnum, CharEnum>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<CharEnum, IntEnum>());

  // An enum is compatible with its exact underlying type but not a mismatched one.
  STATIC_REQUIRE(abi_detail::__abi_compatible<CharEnum, char>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<CharEnum, int>());
  STATIC_REQUIRE(abi_detail::__abi_compatible<IntEnum, int>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<IntEnum, char>());

  // --- Function pointers --------------------------------------------------------
  STATIC_REQUIRE(abi_detail::__abi_compatible<int (*)(Foo_st*), int (*)(FooStruct)>());

  // Return type, arity, and argument types must all match.
  STATIC_REQUIRE(!abi_detail::__abi_compatible<int (*)(int), float (*)(int)>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<int (*)(int), int (*)(int, int)>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<int (*)(int), int (*)(long)>());

  // void return and no-argument functions.
  STATIC_REQUIRE(abi_detail::__abi_compatible<void (*)(), void (*)()>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<void (*)(), int (*)()>());

  // Per-argument ABI compatibility recurses (enum / opaque-pointer arguments).
  STATIC_REQUIRE(abi_detail::__abi_compatible<void (*)(FooEnum), void (*)(Foo_enum)>());
  STATIC_REQUIRE(abi_detail::__abi_compatible<int (*)(Foo_st*, FooEnum), int (*)(FooStruct, Foo_enum)>());
  STATIC_REQUIRE(!abi_detail::__abi_compatible<void (*)(CharEnum), void (*)(IntEnum)>());

  // A function type is not compatible with a non-function type.
  STATIC_REQUIRE(!abi_detail::__abi_compatible<int (*)(int), int*>());
}
