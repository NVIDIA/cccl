//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo(dabayer): nvrtc doesn't support non-trivial types as static data members without -default-device, fails with:
//   A class static data member with non-const type is considered a host variable, and host variables are not allowed in
//   JIT mode. Consider using -default-device flag to process such data members as __device__ variables in JIT mode

// constant_wrapper

//  template<constexpr-param T>
//    friend constexpr auto operator+(T) noexcept -> constant_wrapper<(+T::value)>
//      { return {}; }
//  template<constexpr-param T>
//    friend constexpr auto operator-(T) noexcept -> constant_wrapper<(-T::value)>
//      { return {}; }
//  template<constexpr-param T>
//    friend constexpr auto operator~(T) noexcept -> constant_wrapper<(~T::value)>
//      { return {}; }
//  template<constexpr-param T>
//    friend constexpr auto operator!(T) noexcept -> constant_wrapper<(!T::value)>
//      { return {}; }
//  template<constexpr-param T>
//    friend constexpr auto operator&(T) noexcept -> constant_wrapper<(&T::value)>
//      { return {}; }
//  template<constexpr-param T>
//    friend constexpr auto operator*(T) noexcept -> constant_wrapper<(*T::value)>
//      { return {}; }

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "helpers.h"
#include "test_macros.h"

// gcc < 14 warns about comparing &value == &value (which is always true).
#if TEST_COMPILER(GCC, <, 14)
TEST_DIAG_SUPPRESS_GCC("-Wtautological-compare")
#endif // TEST_COMPILER(GCC, <, 14)

struct WithOps
{
  int value;

  TEST_FUNC constexpr WithOps(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator+(WithOps w)
  {
    return WithOps{+w.value};
  }
  TEST_FUNC friend constexpr auto operator-(WithOps w)
  {
    return WithOps{-w.value};
  }
  TEST_FUNC friend constexpr auto operator~(WithOps w)
  {
    return WithOps{~w.value};
  }
  TEST_FUNC friend constexpr auto operator!(WithOps w)
  {
    return WithOps{!w.value};
  }
  TEST_FUNC friend constexpr auto operator&(WithOps w)
  {
    return WithOps{w.value + 42};
  }
  TEST_FUNC friend constexpr auto operator*(WithOps w)
  {
    return WithOps{w.value - 42};
  }
};

struct OpsReturnNonStructural
{
  int value;

  TEST_FUNC constexpr OpsReturnNonStructural(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator+(OpsReturnNonStructural o)
  {
    return NonStructural{+o.value};
  }
  TEST_FUNC friend constexpr auto operator-(OpsReturnNonStructural o)
  {
    return NonStructural{-o.value};
  }
  TEST_FUNC friend constexpr auto operator~(OpsReturnNonStructural o)
  {
    return NonStructural{~o.value};
  }
  TEST_FUNC friend constexpr auto operator!(OpsReturnNonStructural o)
  {
    return NonStructural{!o.value};
  }
  TEST_FUNC friend constexpr auto operator&(OpsReturnNonStructural o)
  {
    return NonStructural{o.value + 42};
  }
  TEST_FUNC friend constexpr auto operator*(OpsReturnNonStructural o)
  {
    return NonStructural{o.value - 42};
  }
};

struct NoOps
{};

template <class T, class = void>
inline constexpr bool HasPlus = false;
template <class T>
inline constexpr bool HasPlus<T, cuda::std::void_t<decltype(+cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasMinus = false;
template <class T>
inline constexpr bool HasMinus<T, cuda::std::void_t<decltype(-cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasBitNot = false;
template <class T>
inline constexpr bool HasBitNot<T, cuda::std::void_t<decltype(~cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasNot = false;
template <class T>
inline constexpr bool HasNot<T, cuda::std::void_t<decltype(!cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasBitAnd = false;
template <class T>
inline constexpr bool HasBitAnd<T, cuda::std::void_t<decltype(&cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasDeref = false;
template <class T>
inline constexpr bool HasDeref<T, cuda::std::void_t<decltype(*cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasNoexceptPlus = false;
template <class T>
inline constexpr bool HasNoexceptPlus<T, cuda::std::enable_if_t<noexcept(+cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasNoexceptMinus = false;
template <class T>
inline constexpr bool HasNoexceptMinus<T, cuda::std::enable_if_t<noexcept(-cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasNoexceptBitNot = false;
template <class T>
inline constexpr bool HasNoexceptBitNot<T, cuda::std::enable_if_t<noexcept(~cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasNoexceptNot = false;
template <class T>
inline constexpr bool HasNoexceptNot<T, cuda::std::enable_if_t<noexcept(!cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasNoexceptBitAnd = false;
template <class T>
inline constexpr bool HasNoexceptBitAnd<T, cuda::std::enable_if_t<noexcept(&cuda::std::declval<T&>())>> = true;

template <class T, class = void>
inline constexpr bool HasNoexceptDeref = false;
template <class T>
inline constexpr bool HasNoexceptDeref<T, cuda::std::enable_if_t<noexcept(*cuda::std::declval<T&>())>> = true;

#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

static_assert(HasPlus<cuda::std::__constant_wrapper<WithOps{42}>>);
static_assert(HasMinus<cuda::std::__constant_wrapper<WithOps{42}>>);
static_assert(HasBitNot<cuda::std::__constant_wrapper<WithOps{42}>>);
static_assert(HasNot<cuda::std::__constant_wrapper<WithOps{42}>>);
static_assert(HasBitAnd<cuda::std::__constant_wrapper<WithOps{42}>>);
static_assert(HasDeref<cuda::std::__constant_wrapper<WithOps{42}>>);

static_assert(HasNoexceptPlus<cuda::std::__constant_wrapper<WithOps{42}>>);
static_assert(HasNoexceptMinus<cuda::std::__constant_wrapper<WithOps{42}>>);
static_assert(HasNoexceptBitNot<cuda::std::__constant_wrapper<WithOps{42}>>);
static_assert(HasNoexceptNot<cuda::std::__constant_wrapper<WithOps{42}>>);
static_assert(HasNoexceptBitAnd<cuda::std::__constant_wrapper<WithOps{42}>>);
static_assert(HasNoexceptDeref<cuda::std::__constant_wrapper<WithOps{42}>>);

#endif // TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

// Old msvc doesn't evaluate noexcept properly.
#if !TEST_COMPILER(MSVC, <, 19, 30)
static_assert(HasNoexceptPlus<cuda::std::__constant_wrapper<42>>);
static_assert(HasNoexceptMinus<cuda::std::__constant_wrapper<42>>);
static_assert(HasNoexceptBitNot<cuda::std::__constant_wrapper<42>>);
static_assert(HasNoexceptNot<cuda::std::__constant_wrapper<42>>);
static_assert(HasNoexceptBitAnd<cuda::std::__constant_wrapper<42>>);
#endif // !TEST_COMPILER(MSVC, <, 19, 30)
static_assert(!HasDeref<cuda::std::__constant_wrapper<42>>);

#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

static_assert(!HasPlus<cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasMinus<cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasBitNot<cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasNot<cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(HasBitAnd<cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasDeref<cuda::std::__constant_wrapper<NoOps{}>>);

// The operators from constant_wrapper do not exist, but they can be implicited converted
// to the underlying type and use its operators instead.
static_assert(HasPlus<cuda::std::__constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(HasMinus<cuda::std::__constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(HasBitNot<cuda::std::__constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(HasNot<cuda::std::__constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(HasBitAnd<cuda::std::__constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(HasDeref<cuda::std::__constant_wrapper<OpsReturnNonStructural{42}>>);

static_assert(!HasNoexceptPlus<cuda::std::__constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(!HasNoexceptMinus<cuda::std::__constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(!HasNoexceptBitNot<cuda::std::__constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(!HasNoexceptNot<cuda::std::__constant_wrapper<OpsReturnNonStructural{42}>>);
static_assert(!HasNoexceptBitAnd<cuda::std::__constant_wrapper<OpsReturnNonStructural{42}>>);
// todo(dabayer): This is failing with MSVC.
#  if !_CCCL_COMPILER(MSVC)
static_assert(!HasNoexceptDeref<cuda::std::__constant_wrapper<OpsReturnNonStructural{42}>>);
#  endif // !_CCCL_COMPILER(MSVC)

#endif // TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

TEST_FUNC constexpr bool test()
{
  {
    // int
    cuda::std::__constant_wrapper<42> cw42{};

    decltype(auto) result = +cw42;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<42>, decltype(result)>);
    static_assert(result == 42);

    decltype(auto) result2 = -cw42;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<-42>, decltype(result2)>);
    static_assert(result2 == -42);

    decltype(auto) result3 = ~cw42;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<~42>, decltype(result3)>);
    static_assert(result3 == ~42);

    decltype(auto) result4 = !cw42;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<!42>, decltype(result4)>);
    static_assert(result4 == !42);

    // todo(dabayer): This is failing with MSVC.
#if !_CCCL_COMPILER(MSVC)
    decltype(auto) result5 = &cw42;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<&cw42.value>, decltype(result5)>);
    static_assert(result5 == &cw42.value);
#endif // !_CCCL_COMPILER(MSVC)
  }

#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

  {
    // WithOps
    cuda::std::__constant_wrapper<WithOps{42}> cwWithOps;

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{42}>> decltype(auto) result = +cwWithOps;
    static_assert(result.__get().value == 42);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{-42}>> decltype(auto) result2 =
      -cwWithOps;
    static_assert(result2.__get().value == -42);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{~42}>> decltype(auto) result3 =
      ~cwWithOps;
    static_assert(result3.__get().value == ~42);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{!42}>> decltype(auto) result4 =
      !cwWithOps;
    static_assert(result4.__get().value == !42);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{84}>> decltype(auto) result5 = &cwWithOps;
    static_assert(result5.__get().value == 84);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{0}>> decltype(auto) result6 = *cwWithOps;
    static_assert(result6.__get().value == 0);
  }

  {
    // Return non-structural type
    // Will use underlying type's runtime operators
    cuda::std::__constant_wrapper<OpsReturnNonStructural{42}> cwOpsReturnNonStructural;

    cuda::std::same_as<NonStructural> decltype(auto) result = +cwOpsReturnNonStructural;
    assert(result.get() == 42);

    cuda::std::same_as<NonStructural> decltype(auto) result2 = -cwOpsReturnNonStructural;
    assert(result2.get() == -42);

    cuda::std::same_as<NonStructural> decltype(auto) result3 = ~cwOpsReturnNonStructural;
    assert(result3.get() == ~42);

    cuda::std::same_as<NonStructural> decltype(auto) result4 = !cwOpsReturnNonStructural;
    assert(result4.get() == !42);

    cuda::std::same_as<NonStructural> decltype(auto) result5 = &cwOpsReturnNonStructural;
    assert(result5.get() == 84);

    // todo(dabayer): This is failing with MSVC.
#  if !_CCCL_COMPILER(MSVC)
    cuda::std::same_as<NonStructural> decltype(auto) result6 = *cwOpsReturnNonStructural;
    assert(result6.get() == 0);
#  endif // !_CCCL_COMPILER(MSVC)
  }

#endif // TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
