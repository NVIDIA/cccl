//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template<ValueType T, size_t N>
//   requires Swappable<T>
//   void
//   swap(T (&a)[N], T (&b)[N]);

#include <cuda/std/__algorithm_>
#include <cuda/std/__memory_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

#if !defined(TEST_COMPILER_NVRTC)
#  include <utility>
#endif // !TEST_COMPILER_NVRTC

struct CopyOnly
{
  __host__ __device__ CopyOnly() {}
  __host__ __device__ CopyOnly(CopyOnly const&) noexcept {}
  __host__ __device__ CopyOnly& operator=(CopyOnly const&)
  {
    return *this;
  }
};

struct NoexceptMoveOnly
{
  __host__ __device__ NoexceptMoveOnly() {}
  __host__ __device__ NoexceptMoveOnly(NoexceptMoveOnly&&) noexcept {}
  __host__ __device__ NoexceptMoveOnly& operator=(NoexceptMoveOnly&&) noexcept
  {
    return *this;
  }
};

struct NotMoveConstructible
{
  __host__ __device__ NotMoveConstructible() {}
  __host__ __device__ NotMoveConstructible& operator=(NotMoveConstructible&&)
  {
    return *this;
  }

private:
  __host__ __device__ NotMoveConstructible(NotMoveConstructible&&);
};

template <class Tp>
__host__ __device__ auto
can_swap_test(int) -> decltype(cuda::std::swap(cuda::std::declval<Tp>(), cuda::std::declval<Tp>()));

template <class Tp>
__host__ __device__ auto can_swap_test(...) -> cuda::std::false_type;

template <class Tp>
__host__ __device__ constexpr bool can_swap()
{
  return cuda::std::is_same<decltype(can_swap_test<Tp>(0)), void>::value;
}

#if TEST_STD_VER >= 2014
__host__ __device__ constexpr bool test_swap_constexpr()
{
  int i[3] = {1, 2, 3};
  int j[3] = {4, 5, 6};
  cuda::std::swap(i, j);
  return i[0] == 4 && i[1] == 5 && i[2] == 6 && j[0] == 1 && j[1] == 2 && j[2] == 3;
}
#endif // TEST_STD_VER >= 2014

__host__ __device__ void test_ambiguous_std()
{
#if !defined(TEST_COMPILER_NVRTC) && !defined(TEST_COMPILER_MSVC_2017)
  // clang-format off
  NV_IF_TARGET(NV_IS_HOST, (
    cuda::std::pair<::std::pair<int, int>, int> i[3] = {};
    cuda::std::pair<::std::pair<int, int>, int> j[3] = {};
    swap(i,j);
  ))
  // clang-format on
#endif // !TEST_COMPILER_NVRTC && !TEST_COMPILER_MSVC_2017
}

int main(int, char**)
{
  {
    int i[3] = {1, 2, 3};
    int j[3] = {4, 5, 6};
    cuda::std::swap(i, j);
    assert(i[0] == 4);
    assert(i[1] == 5);
    assert(i[2] == 6);
    assert(j[0] == 1);
    assert(j[1] == 2);
    assert(j[2] == 3);
  }
  {
    cuda::std::unique_ptr<int> i[3];
    for (int k = 0; k < 3; ++k)
    {
      i[k].reset(new int(k + 1));
    }
    cuda::std::unique_ptr<int> j[3];
    for (int k = 0; k < 3; ++k)
    {
      j[k].reset(new int(k + 4));
    }
    cuda::std::swap(i, j);
    assert(*i[0] == 4);
    assert(*i[1] == 5);
    assert(*i[2] == 6);
    assert(*j[0] == 1);
    assert(*j[1] == 2);
    assert(*j[2] == 3);
  }
  {
    using CA = CopyOnly[42];
    using MA = NoexceptMoveOnly[42];
    using NA = NotMoveConstructible[42];
    static_assert(can_swap<CA&>(), "");
    static_assert(can_swap<MA&>(), "");
    static_assert(!can_swap<NA&>(), "");

    CA ca;
    MA ma;
    static_assert(!noexcept(cuda::std::swap(ca, ca)), "");
    static_assert(noexcept(cuda::std::swap(ma, ma)), "");
  }

#if TEST_STD_VER >= 2014
  static_assert(test_swap_constexpr(), "");
#endif // TEST_STD_VER >= 2014

  test_ambiguous_std();

  return 0;
}
