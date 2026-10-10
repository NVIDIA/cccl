//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class T>
//   requires MoveAssignable<T> && MoveConstructible<T>
//   void
//   swap(T& a, T& b);

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
#  include <memory>
#  include <utility>
#endif // !TEST_COMPILER(NVRTC)

struct CopyOnly
{
  TEST_FUNC CopyOnly() {}
  TEST_FUNC CopyOnly(CopyOnly const&) noexcept {}
  TEST_FUNC CopyOnly& operator=(CopyOnly const&)
  {
    return *this;
  }
};

struct MoveOnly
{
  TEST_FUNC MoveOnly() {}
  TEST_FUNC MoveOnly(MoveOnly&&) {}
  TEST_FUNC MoveOnly& operator=(MoveOnly&&) noexcept
  {
    return *this;
  }
};

struct NoexceptMoveOnly
{
  TEST_FUNC NoexceptMoveOnly() {}
  TEST_FUNC NoexceptMoveOnly(NoexceptMoveOnly&&) noexcept {}
  TEST_FUNC NoexceptMoveOnly& operator=(NoexceptMoveOnly&&) noexcept
  {
    return *this;
  }
};

struct NotMoveConstructible
{
  TEST_FUNC NotMoveConstructible& operator=(NotMoveConstructible&&)
  {
    return *this;
  }

private:
  TEST_FUNC NotMoveConstructible(NotMoveConstructible&&);
};

struct NotMoveAssignable
{
  TEST_FUNC NotMoveAssignable(NotMoveAssignable&&);

private:
  TEST_FUNC NotMoveAssignable& operator=(NotMoveAssignable&&);
};

template <class Tp>
TEST_FUNC auto can_swap_test(int) -> decltype(cuda::std::swap(cuda::std::declval<Tp>(), cuda::std::declval<Tp>()));

template <class Tp>
TEST_FUNC auto can_swap_test(...) -> cuda::std::false_type;

template <class Tp>
TEST_FUNC constexpr bool can_swap()
{
  return cuda::std::is_same<decltype(can_swap_test<Tp>(0)), void>::value;
}

TEST_FUNC constexpr bool test_swap_constexpr()
{
  int i = 1;
  int j = 2;
  cuda::std::swap(i, j);
  return i == 2 && j == 1;
}

template <class T>
struct swap_with_friend
{
  TEST_FUNC friend void swap(swap_with_friend&, swap_with_friend&) {}
};

template <typename T>
TEST_FUNC void test_ambiguous_std()
{
  // clang-format off
  NV_IF_TARGET(NV_IS_HOST, (
    // fully qualified calls
    {
      T i = {};
      T j = {};
      cuda::std::swap(i,j);
    }
  ))
#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (
    {
      T i = {};
      T j = {};
      std::swap(i,j);
    }
  ))
#endif // !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (
    // ADL calls
    {
      T i = {};
      T j = {};
      swap(i,j);
    }
  ))
#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (
    {
      T i = {};
      T j = {};
      using cuda::std::swap;
      swap(i,j);
    }
    {
      T i = {};
      T j = {};
      using std::swap;
      swap(i,j);
    }
    {
      T i = {};
      T j = {};
      using std::swap;
      using cuda::std::swap;
      swap(i,j);
    }
  ))
  // clang-format on
#endif // !TEST_COMPILER(NVRTC)
}

int main(int, char**)
{
  {
    int i = 1;
    int j = 2;
    cuda::std::swap(i, j);
    assert(i == 2);
    assert(j == 1);
  }
#if !_CCCL_TILE_COMPILATION() // dynamic memory allocation with non-placement ::operator new is unsupported in tile code
  {
    cuda::std::unique_ptr<int> i(new int(1));
    cuda::std::unique_ptr<int> j(new int(2));
    cuda::std::swap(i, j);
    assert(*i == 2);
    assert(*j == 1);
  }
#endif // !_CCCL_TILE_COMPILATION()
  {
    // test that the swap
    static_assert(can_swap<CopyOnly&>());
    static_assert(can_swap<MoveOnly&>());
    static_assert(can_swap<NoexceptMoveOnly&>());

    static_assert(!can_swap<NotMoveConstructible&>());
    static_assert(!can_swap<NotMoveAssignable&>());

    CopyOnly c;
    MoveOnly m;
    NoexceptMoveOnly nm;
    static_assert(!noexcept(cuda::std::swap(c, c)));
    static_assert(!noexcept(cuda::std::swap(m, m)));
    static_assert(noexcept(cuda::std::swap(nm, nm)));
  }

  static_assert(test_swap_constexpr());

  test_ambiguous_std<cuda::std::pair<int, int>>(); // has cuda::std::swap overload
#if !TEST_COMPILER(NVRTC)
  test_ambiguous_std<::std::pair<int, int>>(); // has std::swap overload
  test_ambiguous_std<cuda::std::pair<::std::pair<int, int>, int>>(); // has std:: and cuda::std as associated namespaces
  test_ambiguous_std<::std::allocator<char>>(); // no std::swap overload

  // Ensure that we do not SFINAE swap out if there is a free function as that will take precedent
  test_ambiguous_std<swap_with_friend<::std::pair<int, int>>>();
#endif // !TEST_COMPILER(NVRTC)

#if !TEST_COMPILER(NVRTC)
  static_assert(cuda::std::is_swappable<cuda::std::pair<::std::pair<int, int>, int>>::value);
  static_assert(cuda::std::is_swappable<swap_with_friend<::std::pair<int, int>>>::value);
#endif // !TEST_COMPILER(NVRTC)

  return 0;
}
