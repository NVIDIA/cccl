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

// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator<=>(L, R) noexcept
//     -> constant_wrapper<(L::value <=> R::value)>
//       { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator<(L, R) noexcept -> constant_wrapper<(L::value < R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator<=(L, R) noexcept -> constant_wrapper<(L::value <= R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator==(L, R) noexcept -> constant_wrapper<(L::value == R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator!=(L, R) noexcept -> constant_wrapper<(L::value != R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator>(L, R) noexcept -> constant_wrapper<(L::value > R::value)>
//     { return {}; }
// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator>=(L, R) noexcept -> constant_wrapper<(L::value >= R::value)>
//     { return {}; }

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "helpers.h"
#include "test_macros.h"

// nvcc 12.0 generates weird input file for host compiler.
#if TEST_CUDA_COMPILER(NVCC, ==, 12, 0)
TEST_DIAG_SUPPRESS_GCC("-Wignored-qualifiers")
#endif // TEST_CUDA_COMPILER(NVCC, ==, 12, 0)

struct WithOps
{
  int value;

  TEST_FUNC constexpr WithOps(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator==(WithOps l, WithOps r)
  {
    return l.value == r.value;
  }
  TEST_FUNC friend constexpr auto operator!=(WithOps l, WithOps r)
  {
    return l.value != r.value;
  }
  TEST_FUNC friend constexpr auto operator<(WithOps l, WithOps r)
  {
    return l.value < r.value;
  }
  TEST_FUNC friend constexpr auto operator<=(WithOps l, WithOps r)
  {
    return l.value <= r.value;
  }
  TEST_FUNC friend constexpr auto operator>=(WithOps l, WithOps r)
  {
    return l.value >= r.value;
  }
  TEST_FUNC friend constexpr auto operator>(WithOps l, WithOps r)
  {
    return l.value > r.value;
  }
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  TEST_FUNC friend constexpr auto operator<=>(WithOps l, WithOps r)
  {
    return l.value <=> r.value;
  }
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

struct OpsReturnNonStructural
{
  int value;

  TEST_FUNC constexpr OpsReturnNonStructural(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator==(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value == r.value ? 1 : 0};
  }
  TEST_FUNC friend constexpr auto operator!=(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value != r.value ? 1 : 0};
  }
  TEST_FUNC friend constexpr auto operator<(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value < r.value ? 1 : 0};
  }
  TEST_FUNC friend constexpr auto operator<=(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value <= r.value ? 1 : 0};
  }
  TEST_FUNC friend constexpr auto operator>=(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value >= r.value ? 1 : 0};
  }
  TEST_FUNC friend constexpr auto operator>(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{l.value > r.value ? 1 : 0};
  }
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  TEST_FUNC friend constexpr auto operator<=>(OpsReturnNonStructural l, OpsReturnNonStructural r)
  {
    return NonStructural{(l.value < r.value) ? -1 : (l.value > r.value) ? 1 : 0};
  }
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

struct NoOps
{};

template <class L, class R, class = void>
inline constexpr bool HasEqual = false;
template <class L, class R>
inline constexpr bool HasEqual<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() == cuda::std::declval<R&>())>> =
  true;

template <class L, class R, class = void>
inline constexpr bool HasNotEqual = false;
template <class L, class R>
inline constexpr bool
  HasNotEqual<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() != cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasLess = false;
template <class L, class R>
inline constexpr bool HasLess<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() < cuda::std::declval<R&>())>> =
  true;

template <class L, class R, class = void>
inline constexpr bool HasLessEqual = false;
template <class L, class R>
inline constexpr bool
  HasLessEqual<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() <= cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasGreater = false;
template <class L, class R>
inline constexpr bool
  HasGreater<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() > cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasGreaterEqual = false;
template <class L, class R>
inline constexpr bool
  HasGreaterEqual<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() >= cuda::std::declval<R&>())>> = true;

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
template <class L, class R, class = void>
inline constexpr bool HasSpaceship = false;
template <class L, class R>
inline constexpr bool
  HasSpaceship<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>() <=> cuda::std::declval<R&>())>> = true;
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

template <class L, class R, class = void>
inline constexpr bool HasNoexceptEqual = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptEqual<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() == cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptNotEqual = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptNotEqual<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() != cuda::std::declval<R&>())>> =
    true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptLess = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptLess<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() < cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptLessEqual = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptLessEqual<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() <= cuda::std::declval<R&>())>> =
    true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptGreater = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptGreater<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() > cuda::std::declval<R&>())>> =
    true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptGreaterEqual = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptGreaterEqual<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() >= cuda::std::declval<R&>())>> =
    true;

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
template <class L, class R, class = void>
inline constexpr bool HasNoexceptSpaceship = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptSpaceship<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>() <=> cuda::std::declval<R&>())>> =
    true;
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

// Concept checks for int comparisons
static_assert(HasEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNotEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasLess<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasLessEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasGreater<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasGreaterEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
static_assert(HasSpaceship<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

// Old msvc doesn't evaluate noexcept properly.
#if !TEST_COMPILER(MSVC, <, 19, 30)
static_assert(HasNoexceptEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptNotEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptLess<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptLessEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptGreater<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
static_assert(HasNoexceptGreaterEqual<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
#  if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
static_assert(HasNoexceptSpaceship<cuda::std::__constant_wrapper<6>, cuda::std::__constant_wrapper<3>>);
#  endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#endif // !TEST_COMPILER(MSVC, <, 19, 30)

#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

// NoOps
static_assert(!HasEqual<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasNotEqual<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasLess<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasLessEqual<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasGreater<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
static_assert(!HasGreaterEqual<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
#  if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
static_assert(!HasSpaceship<cuda::std::__constant_wrapper<NoOps{}>, cuda::std::__constant_wrapper<NoOps{}>>);
#  endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

// Concept checks for WithOps comparisons
static_assert(HasNoexceptEqual<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptNotEqual<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptLess<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptLessEqual<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(HasNoexceptGreater<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
static_assert(
  HasNoexceptGreaterEqual<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>);
#  if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
static_assert(
  !HasNoexceptSpaceship<cuda::std::__constant_wrapper<WithOps{6}>, cuda::std::__constant_wrapper<WithOps{3}>>,
  "strong_ordering is not a structural type, so the call falls back to runtime implicit conversion and "
  "operator<=>, which is noexcept(false)");
#  endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

// clang-format off
// Non-structural types use implicit conversion to underlying type
static_assert(HasEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasNotEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasLess<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasLessEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasGreater<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(HasGreaterEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);

static_assert(!HasNoexceptEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptNotEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptLess<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptLessEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptGreater<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
static_assert(!HasNoexceptGreaterEqual<cuda::std::__constant_wrapper<OpsReturnNonStructural{6}>, cuda::std::__constant_wrapper<OpsReturnNonStructural{3}>>);
// clang-format on

#endif // TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

TEST_FUNC constexpr bool test()
{
  {
    // int comparisons: 6 vs 3 - returns constant_wrapper<bool_value>
    cuda::std::__constant_wrapper<6> cw6{};
    cuda::std::__constant_wrapper<3> cw3{};

    [[maybe_unused]] decltype(auto) equal = cw6 == cw3;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<false>, decltype(equal)>);
// nvcc 13.3 fails to produce correct input file for host compiler. See nvbug 6249821.
#if _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) || _CCCL_TILE_COMPILATION()
    static_assert(!decltype(equal)::value);
#else // ^^^ _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^
    static_assert(!static_cast<bool>(equal));
#endif // ^^^ !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^

    [[maybe_unused]] decltype(auto) not_equal = cw6 != cw3;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<true>, decltype(not_equal)>);
// nvcc 13.3 fails to produce correct input file for host compiler. See nvbug 6249821.
#if _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) || _CCCL_TILE_COMPILATION()
    static_assert(decltype(not_equal)::value);
#else // ^^^ _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^
    static_assert(static_cast<bool>(not_equal));
#endif // ^^^ !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^

    [[maybe_unused]] decltype(auto) less = cw6 < cw3;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<false>, decltype(less)>);
// nvcc 13.3 fails to produce correct input file for host compiler. See nvbug 6249821.
#if _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) || _CCCL_TILE_COMPILATION()
    static_assert(!decltype(less)::value);
#else // ^^^ _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^
    static_assert(!static_cast<bool>(less));
#endif // ^^^ !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^

    [[maybe_unused]] decltype(auto) less_equal = cw6 <= cw3;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<false>, decltype(less_equal)>);
// nvcc 13.3 fails to produce correct input file for host compiler. See nvbug 6249821.
#if _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) || _CCCL_TILE_COMPILATION()
    static_assert(!decltype(less_equal)::value);
#else // ^^^ _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^
    static_assert(!static_cast<bool>(less_equal));
#endif // ^^^ !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^

    [[maybe_unused]] decltype(auto) greater = cw6 > cw3;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<true>, decltype(greater)>);
// nvcc 13.3 fails to produce correct input file for host compiler. See nvbug 6249821.
#if _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) || _CCCL_TILE_COMPILATION()
    static_assert(decltype(greater)::value);
#else // ^^^ _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^
    static_assert(static_cast<bool>(greater));
#endif // ^^^ !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^

    [[maybe_unused]] decltype(auto) greater_equal = cw6 >= cw3;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<true>, decltype(greater_equal)>);
// nvcc 13.3 fails to produce correct input file for host compiler. See nvbug 6249821.
#if _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) || _CCCL_TILE_COMPILATION()
    static_assert(decltype(greater_equal)::value);
#else // ^^^ _CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^
    static_assert(static_cast<bool>(greater_equal));
#endif // ^^^ !_CCCL_CUDA_COMPILER(NVCC, ==, 13, 3) ^^^

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    // strong_ordering is not a structural type
    cuda::std::same_as<cuda::std::strong_ordering> decltype(auto) spaceship = cw6 <=> cw3;
    assert(spaceship == cuda::std::strong_ordering::greater);
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  }

  {
    // int comparisons: equal values
    cuda::std::__constant_wrapper<3> cw3a{};
    cuda::std::__constant_wrapper<3> cw3b{};

    decltype(auto) equal = cw3a == cw3b;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<true>, decltype(equal)>);
    static_assert(static_cast<bool>(equal));

    decltype(auto) not_equal = cw3a != cw3b;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<false>, decltype(not_equal)>);
    static_assert(!static_cast<bool>(not_equal));

    decltype(auto) less = cw3a < cw3b;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<false>, decltype(less)>);
    static_assert(!static_cast<bool>(less));

    decltype(auto) less_equal = cw3a <= cw3b;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<true>, decltype(less_equal)>);
    static_assert(static_cast<bool>(less_equal));

    decltype(auto) greater = cw3a >= cw3b;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<true>, decltype(greater)>);
    static_assert(static_cast<bool>(greater));

    decltype(auto) greater_cmp = cw3a > cw3b;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<false>, decltype(greater_cmp)>);
    static_assert(!static_cast<bool>(greater_cmp));

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    cuda::std::same_as<cuda::std::strong_ordering> decltype(auto) spaceship = cw3a <=> cw3b;
    assert(spaceship == cuda::std::strong_ordering::equal);
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  }

#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

  {
    // WithOps comparisons - returns constant_wrapper<bool_value>
    cuda::std::__constant_wrapper<WithOps{6}> cwWithOps6;
    cuda::std::__constant_wrapper<WithOps{3}> cwWithOps3;

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) equal = cwWithOps6 == cwWithOps3;
    static_assert(!static_cast<bool>(equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) not_equal = cwWithOps6 != cwWithOps3;
    static_assert(static_cast<bool>(not_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) less = cwWithOps6 < cwWithOps3;
    static_assert(!static_cast<bool>(less));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) less_equal = cwWithOps6 <= cwWithOps3;
    static_assert(!static_cast<bool>(less_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) greater = cwWithOps6 > cwWithOps3;
    static_assert(static_cast<bool>(greater));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) greater_equal = cwWithOps6 >= cwWithOps3;
    static_assert(static_cast<bool>(greater_equal));

#  if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    cuda::std::same_as<cuda::std::strong_ordering> decltype(auto) spaceship = cwWithOps6 <=> cwWithOps3;
    assert(spaceship == cuda::std::strong_ordering::greater);
#  endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  }

  {
    // WithOps comparisons: equal values
    cuda::std::__constant_wrapper<WithOps{3}> cwWithOps3a;
    cuda::std::__constant_wrapper<WithOps{3}> cwWithOps3b;

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) equal = cwWithOps3a == cwWithOps3b;
    static_assert(static_cast<bool>(equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) not_equal = cwWithOps3a != cwWithOps3b;
    static_assert(!static_cast<bool>(not_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) less = cwWithOps3a < cwWithOps3b;
    static_assert(!static_cast<bool>(less));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) less_equal = cwWithOps3a <= cwWithOps3b;
    static_assert(static_cast<bool>(less_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) greater_equal = cwWithOps3a >= cwWithOps3b;
    static_assert(static_cast<bool>(greater_equal));

    cuda::std::same_as<cuda::std::__constant_wrapper<false>> decltype(auto) greater = cwWithOps3a > cwWithOps3b;
    static_assert(!static_cast<bool>(greater));

#  if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    cuda::std::same_as<cuda::std::strong_ordering> decltype(auto) spaceship = cwWithOps3a <=> cwWithOps3b;
    assert(spaceship == cuda::std::strong_ordering::equal);
#  endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  }

  {
    // Non-structural return types use implicit conversion
    cuda::std::__constant_wrapper<OpsReturnNonStructural{6}> cwOpt6;
    cuda::std::__constant_wrapper<OpsReturnNonStructural{3}> cwOpt3;

    cuda::std::same_as<NonStructural> decltype(auto) equal = cwOpt6 == cwOpt3;
    assert(equal.get() == 0);

    cuda::std::same_as<NonStructural> decltype(auto) not_equal = cwOpt6 != cwOpt3;
    assert(not_equal.get() == 1);

    cuda::std::same_as<NonStructural> decltype(auto) less = cwOpt6 < cwOpt3;
    assert(less.get() == 0);

    cuda::std::same_as<NonStructural> decltype(auto) less_equal = cwOpt6 <= cwOpt3;
    assert(less_equal.get() == 0);

    cuda::std::same_as<NonStructural> decltype(auto) greater = cwOpt6 > cwOpt3;
    assert(greater.get() == 1);

    cuda::std::same_as<NonStructural> decltype(auto) greater_equal = cwOpt6 >= cwOpt3;
    assert(greater_equal.get() == 1);

#  if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    cuda::std::same_as<NonStructural> decltype(auto) spaceship = cwOpt6 <=> cwOpt3;
    assert(spaceship.get() == 1);
#  endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  }

#endif // TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

  {
    // Mix with runtime param: these operators are not used (built-in operators)
    cuda::std::__constant_wrapper<6> cw6{};
    int i = 3;

    decltype(auto) equal = cw6 == i;
    static_assert(cuda::std::same_as<bool, decltype(equal)>);
    assert(!equal);

    decltype(auto) not_equal = cw6 != i;
    static_assert(cuda::std::same_as<bool, decltype(not_equal)>);
    assert(not_equal);

    decltype(auto) less = cw6 < i;
    static_assert(cuda::std::same_as<bool, decltype(less)>);
    assert(!less);

    decltype(auto) less_equal = cw6 <= i;
    static_assert(cuda::std::same_as<bool, decltype(less_equal)>);
    assert(!less_equal);

    decltype(auto) greater = cw6 > i;
    static_assert(cuda::std::same_as<bool, decltype(greater)>);
    assert(greater);

    decltype(auto) greater_equal = cw6 >= i;
    static_assert(cuda::std::same_as<bool, decltype(greater_equal)>);
    assert(greater_equal);

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    cuda::std::same_as<cuda::std::strong_ordering> decltype(auto) spaceship = cw6 <=> i;
    assert(spaceship == cuda::std::strong_ordering::greater);
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  }

  {
    // with integral_constant
    cuda::std::__constant_wrapper<6> cw6{};
    cuda::std::integral_constant<int, 3> ic3{};

    decltype(auto) equal = cw6 == ic3;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<false>, decltype(equal)>);
    static_assert(!static_cast<bool>(equal));

    decltype(auto) not_equal = cw6 != ic3;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<true>, decltype(not_equal)>);
    static_assert(static_cast<bool>(not_equal));

    decltype(auto) less = cw6 < ic3;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<false>, decltype(less)>);
    static_assert(!static_cast<bool>(less));

    decltype(auto) less_equal = cw6 <= ic3;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<false>, decltype(less_equal)>);
    static_assert(!static_cast<bool>(less_equal));

    decltype(auto) greater = cw6 > ic3;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<true>, decltype(greater)>);
    static_assert(static_cast<bool>(greater));

    decltype(auto) greater_equal = cw6 >= ic3;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<true>, decltype(greater_equal)>);
    static_assert(static_cast<bool>(greater_equal));

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    // strong_ordering is not a structural type
    cuda::std::same_as<cuda::std::strong_ordering> decltype(auto) spaceship = cw6 <=> ic3;
    assert(spaceship == cuda::std::strong_ordering::greater);
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
