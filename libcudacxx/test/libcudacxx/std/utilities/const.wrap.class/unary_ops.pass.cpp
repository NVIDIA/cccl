//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// gcc-10 segfaults with any use of constant_wrapper, gcc-11 fails to evaluate:
//   typename decltype(__cw_fixed_value(_Xp))::type
// UNSUPPORTED: gcc-10 || gcc-11

// nvcc 12.0 segfaults.
// UNSUPPORTED: nvcc-12.0

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
// UNSUPPORTED: nvrtc

// REQUIRES: !c++17

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
#include <cuda/std/utility>

#include "helpers.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(20094) // a host member cannot be directly read in a __device__/__global__ function

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

template <class T>
concept HasPlus = requires(T t) {
  { +t };
};

template <class T>
concept HasMinus = requires(T t) {
  { -t };
};

template <class T>
concept HasBitNot = requires(T t) {
  { ~t };
};

template <class T>
concept HasNot = requires(T t) {
  { !t };
};

template <class T>
concept HasBitAnd = requires(T t) {
  { &t };
};

template <class T>
concept HasDeref = requires(T t) {
  { *t };
};

template <class T>
concept HasNoexceptPlus = requires(T t) {
  { +t } noexcept;
};

template <class T>
concept HasNoexceptMinus = requires(T t) {
  { -t } noexcept;
};

template <class T>
concept HasNoexceptBitNot = requires(T t) {
  { ~t } noexcept;
};

template <class T>
concept HasNoexceptNot = requires(T t) {
  { !t } noexcept;
};

template <class T>
concept HasNoexceptBitAnd = requires(T t) {
  { &t } noexcept;
};

template <class T>
concept HasNoexceptDeref = requires(T t) {
  { *t } noexcept;
};

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

static_assert(HasNoexceptPlus<cuda::std::__constant_wrapper<42>>);
static_assert(HasNoexceptMinus<cuda::std::__constant_wrapper<42>>);
static_assert(HasNoexceptBitNot<cuda::std::__constant_wrapper<42>>);
static_assert(HasNoexceptNot<cuda::std::__constant_wrapper<42>>);
static_assert(HasNoexceptBitAnd<cuda::std::__constant_wrapper<42>>);
static_assert(!HasDeref<cuda::std::__constant_wrapper<42>>);

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
static_assert(!HasNoexceptDeref<cuda::std::__constant_wrapper<OpsReturnNonStructural{42}>>);

TEST_FUNC constexpr bool test()
{
  {
    // int
    cuda::std::__constant_wrapper<42> cw42;

    cuda::std::same_as<cuda::std::__constant_wrapper<42>> decltype(auto) result = +cw42;
    static_assert(result == 42);

    cuda::std::same_as<cuda::std::__constant_wrapper<-42>> decltype(auto) result2 = -cw42;
    static_assert(result2 == -42);

    cuda::std::same_as<cuda::std::__constant_wrapper<~42>> decltype(auto) result3 = ~cw42;
    static_assert(result3 == ~42);

    cuda::std::same_as<cuda::std::__constant_wrapper<!42>> decltype(auto) result4 = !cw42;
    static_assert(result4 == !42);

    // gcc < 13 fails this test with error:
    //  the address of 'cuda::std::__4::__cw_fixed_value<int>{42}' is not a valid template argument
#if !_CCCL_COMPILER(GCC, <, 13)
    cuda::std::same_as<cuda::std::__constant_wrapper<&cw42.value>> decltype(auto) result5 = &cw42;
    static_assert(result5 == &cw42.value);
#endif // !_CCCL_COMPILER(GCC, <, 13)
  }

  {
    // WithOps
    cuda::std::__constant_wrapper<WithOps{42}> cwWithOps;

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{42}>> decltype(auto) result = +cwWithOps;
    static_assert(result.value.value == 42);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{-42}>> decltype(auto) result2 =
      -cwWithOps;
    static_assert(result2.value.value == -42);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{~42}>> decltype(auto) result3 =
      ~cwWithOps;
    static_assert(result3.value.value == ~42);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{!42}>> decltype(auto) result4 =
      !cwWithOps;
    static_assert(result4.value.value == !42);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{84}>> decltype(auto) result5 = &cwWithOps;
    static_assert(result5.value.value == 84);

    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<WithOps{0}>> decltype(auto) result6 = *cwWithOps;
    static_assert(result6.value.value == 0);
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

    cuda::std::same_as<NonStructural> decltype(auto) result6 = *cwOpsReturnNonStructural;
    assert(result6.get() == 0);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
