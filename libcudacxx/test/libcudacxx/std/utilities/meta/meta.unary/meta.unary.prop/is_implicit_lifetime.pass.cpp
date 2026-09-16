//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <type_traits>

// template<class T> struct is_implicit_lifetime;

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/expected>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#include "test_macros.h"

enum Enum
{
  EV
};
enum SignedEnum : signed int
{
};
enum UnsignedEnum : unsigned int
{
};

enum class EnumClass
{
  EV
};
enum class SignedEnumClass : signed int
{
};
enum class UnsignedEnumClass : unsigned int
{
};

struct EmptyStruct
{};
struct IncompleteStruct;

struct NoEligibleTrivialConstructor
{
  TEST_FUNC NoEligibleTrivialConstructor() {};
  TEST_FUNC NoEligibleTrivialConstructor(const NoEligibleTrivialConstructor&) {}
  TEST_FUNC NoEligibleTrivialConstructor(NoEligibleTrivialConstructor&&) {}
};

struct OnlyDefaultConstructorIsTrivial
{
  OnlyDefaultConstructorIsTrivial() = default;
  TEST_FUNC OnlyDefaultConstructorIsTrivial(const OnlyDefaultConstructorIsTrivial&) {}
  TEST_FUNC OnlyDefaultConstructorIsTrivial(OnlyDefaultConstructorIsTrivial&&) {}
};

struct AllConstructorsAreTrivial
{
  AllConstructorsAreTrivial()                                 = default;
  AllConstructorsAreTrivial(const AllConstructorsAreTrivial&) = default;
  AllConstructorsAreTrivial(AllConstructorsAreTrivial&&)      = default;
};

struct InheritedNoEligibleTrivialConstructor : NoEligibleTrivialConstructor
{
  using NoEligibleTrivialConstructor::NoEligibleTrivialConstructor;
};

struct InheritedOnlyDefaultConstructorIsTrivial : OnlyDefaultConstructorIsTrivial
{
  using OnlyDefaultConstructorIsTrivial::OnlyDefaultConstructorIsTrivial;
};

struct InheritedAllConstructorsAreTrivial : AllConstructorsAreTrivial
{
  using AllConstructorsAreTrivial::AllConstructorsAreTrivial;
};

struct UserDeclaredDestructor
{
  ~UserDeclaredDestructor() = default;
};

struct UserProvidedDestructor
{
  TEST_FUNC ~UserProvidedDestructor() {}
};

struct UserDeletedDestructorInAggregate
{
  ~UserDeletedDestructorInAggregate() = delete;
};

struct UserDeletedDestructorInNonAggregate
{
  virtual void NonAggregate();
  ~UserDeletedDestructorInNonAggregate() = delete;
};

struct DeletedDestructorViaBaseInAggregate : UserDeletedDestructorInAggregate
{};
struct DeletedDestructorViaBaseInNonAggregate : UserDeletedDestructorInNonAggregate
{};

#if TEST_STD_VER >= 2020
template <bool B>
struct ConstrainedUserDeclaredDefaultConstructor
{
  ConstrainedUserDeclaredDefaultConstructor()
    requires B
  = default;
  TEST_FUNC ConstrainedUserDeclaredDefaultConstructor(const ConstrainedUserDeclaredDefaultConstructor&) {}
};

template <bool B>
struct ConstrainedUserProvidedDestructor
{
  ~ConstrainedUserProvidedDestructor() = default;
  TEST_FUNC ~ConstrainedUserProvidedDestructor()
    requires B
  {}
};
#else // ^^^ TEST_STD_VER >= 2020 ^^^ / vvv TEST_STD_VER < 2020 vvv
template <bool B>
struct ConstrainedUserDeclaredDefaultConstructor
{
  template <bool B2 = B, cuda::std::enable_if_t<B2, int> = 0>
  TEST_FUNC ConstrainedUserDeclaredDefaultConstructor() {};
  TEST_FUNC ConstrainedUserDeclaredDefaultConstructor(const ConstrainedUserDeclaredDefaultConstructor&) {}
};
template <>
struct ConstrainedUserDeclaredDefaultConstructor<true>
{
  ConstrainedUserDeclaredDefaultConstructor() = default;
  TEST_FUNC ConstrainedUserDeclaredDefaultConstructor(const ConstrainedUserDeclaredDefaultConstructor&) {}
};

// We can't emulate ConstrainedUserProvidedDestructor in C++17
#endif // ^^^ TEST_STD_VER < 2020 ^^^

#if TEST_COMPILER(CLANG)
struct StructWithFlexibleArrayMember
{
  int arr[];
};
#endif // TEST_COMPILER(CLANG)

struct StructWithZeroSizedArray
{
  int arr[0];
};

// Test implicit-lifetime type
template <typename T, bool Expected>
TEST_FUNC constexpr void test_is_implicit_lifetime()
{
#if defined(_CCCL_BUILTIN_IS_IMPLICIT_LIFETIME)
  assert(cuda::std::is_implicit_lifetime<T>::value == Expected);
  assert(cuda::std::is_implicit_lifetime_v<T> == Expected);
#endif // defined(_CCCL_BUILTIN_IS_IMPLICIT_LIFETIME)
}

// Test pointer, reference, array, etc. types
template <typename T>
TEST_FUNC constexpr void test_is_implicit_lifetime()
{
  test_is_implicit_lifetime<T, true>();

  // cv-qualified
  test_is_implicit_lifetime<const T, true>();
  test_is_implicit_lifetime<volatile T, true>();

  test_is_implicit_lifetime<T&, false>();
  test_is_implicit_lifetime<T&&, false>();

  // Pointer types
  test_is_implicit_lifetime<T*, true>();

  // Arrays
  test_is_implicit_lifetime<T[], true>();
  test_is_implicit_lifetime<T[94], true>();
}

TEST_FUNC constexpr bool test()
{
  // Standard fundamental C++ types

  test_is_implicit_lifetime<cuda::std::nullptr_t, true>();

  test_is_implicit_lifetime<void, false>();
  test_is_implicit_lifetime<const void, false>();
  test_is_implicit_lifetime<volatile void, false>();

  test_is_implicit_lifetime<signed char>();
  test_is_implicit_lifetime<signed short>();
  test_is_implicit_lifetime<signed int>();
  test_is_implicit_lifetime<signed long>();
  test_is_implicit_lifetime<signed long long>();
#if _CCCL_HAS_INT128()
  test_is_implicit_lifetime<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_is_implicit_lifetime<unsigned char>();
  test_is_implicit_lifetime<unsigned short>();
  test_is_implicit_lifetime<unsigned int>();
  test_is_implicit_lifetime<unsigned long>();
  test_is_implicit_lifetime<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_is_implicit_lifetime<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  test_is_implicit_lifetime<float>();
  test_is_implicit_lifetime<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_is_implicit_lifetime<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()

  test_is_implicit_lifetime<Enum>();
  test_is_implicit_lifetime<SignedEnum>();
  test_is_implicit_lifetime<UnsignedEnum>();

  test_is_implicit_lifetime<EnumClass>();
  test_is_implicit_lifetime<SignedEnumClass>();
  test_is_implicit_lifetime<UnsignedEnumClass>();

  test_is_implicit_lifetime<void(), false>();
  test_is_implicit_lifetime<void() &, false>();
  test_is_implicit_lifetime<void() const, false>();
  test_is_implicit_lifetime<void (&)(), false>();
  test_is_implicit_lifetime<void (*)(), true>();

  // Implicit-lifetime class types

  test_is_implicit_lifetime<EmptyStruct>();
  test_is_implicit_lifetime<int EmptyStruct::*, true>(); // Pointer-to-member
  test_is_implicit_lifetime<int (EmptyStruct::*)(), true>();
  test_is_implicit_lifetime<int (EmptyStruct::*)() const, true>();
  test_is_implicit_lifetime<int (EmptyStruct::*)() &, true>();
  test_is_implicit_lifetime<int (EmptyStruct::*)() &&, true>();

  test_is_implicit_lifetime<IncompleteStruct[], true>();
  test_is_implicit_lifetime<IncompleteStruct[82], true>();

  test_is_implicit_lifetime<UserDeclaredDestructor>();

  test_is_implicit_lifetime<UserProvidedDestructor, false>();

  test_is_implicit_lifetime<NoEligibleTrivialConstructor, false>();

  test_is_implicit_lifetime<OnlyDefaultConstructorIsTrivial, true>();

  test_is_implicit_lifetime<AllConstructorsAreTrivial, true>();

  test_is_implicit_lifetime<InheritedNoEligibleTrivialConstructor, false>();

  test_is_implicit_lifetime<InheritedOnlyDefaultConstructorIsTrivial, true>();

  test_is_implicit_lifetime<InheritedAllConstructorsAreTrivial, true>();

  test_is_implicit_lifetime<UserDeletedDestructorInAggregate, true>();

  test_is_implicit_lifetime<UserDeletedDestructorInNonAggregate, false>();

  test_is_implicit_lifetime<DeletedDestructorViaBaseInAggregate, true>();

  test_is_implicit_lifetime<DeletedDestructorViaBaseInNonAggregate, false>();

  test_is_implicit_lifetime<ConstrainedUserDeclaredDefaultConstructor<true>, true>();
#if !TEST_COMPILER(GCC, <, 16, 2) // This is https://gcc.gnu.org/bugzilla/show_bug.cgi?id=126007
  test_is_implicit_lifetime<ConstrainedUserDeclaredDefaultConstructor<false>, false>();
#endif // !TEST_COMPILER(GCC, <, 16, 2)

#if TEST_STD_VER >= 2020
  test_is_implicit_lifetime<ConstrainedUserProvidedDestructor<true>, false>();
  test_is_implicit_lifetime<ConstrainedUserProvidedDestructor<false>, true>();
#endif // TEST_STD_VER >= 2020

#if TEST_COMPILER(CLANG)
  test_is_implicit_lifetime<StructWithFlexibleArrayMember, true>();
#endif // TEST_COMPILER(CLANG)

  test_is_implicit_lifetime<StructWithZeroSizedArray, true>();

  // C++ standard library types

  // These types are guaranteed to be implicit-lifetime.
  test_is_implicit_lifetime<cuda::std::expected<int, float>>();
  test_is_implicit_lifetime<cuda::std::optional<float>>();
  test_is_implicit_lifetime<cuda::std::variant<float, int>>();

  test_is_implicit_lifetime<cuda::std::pair<int, float>>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
