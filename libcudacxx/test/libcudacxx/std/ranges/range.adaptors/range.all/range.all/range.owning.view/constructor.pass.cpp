//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// owning_view() requires default_initializable<R> = default;
// constexpr owning_view(R&& t);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct DefaultConstructible
{
  int i;
  __host__ __device__ constexpr explicit DefaultConstructible(int j = 42) noexcept(false)
      : i(j)
  {}
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct NotDefaultConstructible
{
  int i;
  __host__ __device__ constexpr explicit NotDefaultConstructible(int j) noexcept(false)
      : i(j)
  {}
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct MoveChecker
{
  int i;
  __host__ __device__ constexpr explicit MoveChecker(int j)
      : i(j)
  {}
  __host__ __device__ constexpr MoveChecker(MoveChecker&& v)
      : i(cuda::std::exchange(v.i, -1))
  {}
  __host__ __device__ MoveChecker& operator=(MoveChecker&&);
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct NoexceptChecker
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

__host__ __device__ constexpr bool test()
{
  {
    using OwningView = cuda::std::ranges::owning_view<DefaultConstructible>;
    static_assert(cuda::std::is_constructible_v<OwningView>);
    static_assert(cuda::std::default_initializable<OwningView>);
    static_assert(cuda::std::movable<OwningView>);
    static_assert(cuda::std::is_trivially_move_constructible_v<OwningView>);
    static_assert(cuda::std::is_trivially_move_assignable_v<OwningView>);
    static_assert(!cuda::std::is_copy_constructible_v<OwningView>);
    static_assert(!cuda::std::is_copy_assignable_v<OwningView>);
    static_assert(!cuda::std::is_constructible_v<OwningView, int>);
    static_assert(!cuda::std::is_constructible_v<OwningView, DefaultConstructible&>);
    static_assert(cuda::std::is_constructible_v<OwningView, DefaultConstructible&&>);
    static_assert(!cuda::std::is_convertible_v<int, OwningView>);
    static_assert(cuda::std::is_convertible_v<DefaultConstructible&&, OwningView>);
    {
      OwningView ov;
      assert(ov.base().i == 42);
    }
    {
      OwningView ov = OwningView(DefaultConstructible(1));
      assert(ov.base().i == 1);
    }
  }
  {
    using OwningView = cuda::std::ranges::owning_view<NotDefaultConstructible>;
    static_assert(!cuda::std::is_constructible_v<OwningView>);
    static_assert(!cuda::std::default_initializable<OwningView>);
    static_assert(cuda::std::movable<OwningView>);
    static_assert(cuda::std::is_trivially_move_constructible_v<OwningView>);
    static_assert(cuda::std::is_trivially_move_assignable_v<OwningView>);
    static_assert(!cuda::std::is_copy_constructible_v<OwningView>);
    static_assert(!cuda::std::is_copy_assignable_v<OwningView>);
    static_assert(!cuda::std::is_constructible_v<OwningView, int>);
    static_assert(!cuda::std::is_constructible_v<OwningView, NotDefaultConstructible&>);
    static_assert(cuda::std::is_constructible_v<OwningView, NotDefaultConstructible&&>);
    static_assert(!cuda::std::is_convertible_v<int, OwningView>);
    static_assert(cuda::std::is_convertible_v<NotDefaultConstructible&&, OwningView>);
    {
      OwningView ov = OwningView(NotDefaultConstructible(1));
      assert(ov.base().i == 1);
    }
  }
  {
    using OwningView = cuda::std::ranges::owning_view<MoveChecker>;
    static_assert(!cuda::std::is_constructible_v<OwningView>);
    static_assert(!cuda::std::default_initializable<OwningView>);
    static_assert(cuda::std::movable<OwningView>);
    static_assert(!cuda::std::is_trivially_move_constructible_v<OwningView>);
    static_assert(!cuda::std::is_trivially_move_assignable_v<OwningView>);
    static_assert(!cuda::std::is_copy_constructible_v<OwningView>);
    static_assert(!cuda::std::is_copy_assignable_v<OwningView>);
    static_assert(!cuda::std::is_constructible_v<OwningView, int>);
    static_assert(!cuda::std::is_constructible_v<OwningView, MoveChecker&>);
    static_assert(cuda::std::is_constructible_v<OwningView, MoveChecker&&>);
    static_assert(!cuda::std::is_convertible_v<int, OwningView>);
    static_assert(cuda::std::is_convertible_v<MoveChecker&&, OwningView>);
    {
      // Check that the constructor does indeed move from the target object.
      auto m        = MoveChecker(42);
      OwningView ov = OwningView(cuda::std::move(m));
      assert(ov.base().i == 42);
      assert(m.i == -1);
    }
  }
  {
    // Check that the defaulted constructors are (not) noexcept when appropriate.

    static_assert(cuda::std::is_nothrow_constructible_v<NoexceptChecker>); // therefore,
    static_assert(cuda::std::is_nothrow_constructible_v<cuda::std::ranges::owning_view<NoexceptChecker>>);

#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 9) && !defined(TEST_COMPILER_MSVC) && !defined(TEST_COMPILER_ICC)
    static_assert(!cuda::std::is_nothrow_constructible_v<DefaultConstructible>); // therefore,
#endif // no broken noexcept
#if !defined(TEST_COMPILER_NVCC) && !defined(TEST_COMPILER_NVRTC) // nvbug3910409
    static_assert(!cuda::std::is_nothrow_constructible_v<cuda::std::ranges::owning_view<DefaultConstructible>>);
#endif

    static_assert(cuda::std::is_nothrow_move_constructible_v<NoexceptChecker>); // therefore,
    static_assert(cuda::std::is_nothrow_move_constructible_v<cuda::std::ranges::owning_view<NoexceptChecker>>);
#if !defined(TEST_COMPILER_ICC) // broken noexcept
    static_assert(!cuda::std::is_nothrow_move_constructible_v<MoveChecker>); // therefore,
    static_assert(!cuda::std::is_nothrow_move_constructible_v<cuda::std::ranges::owning_view<MoveChecker>>);
#endif // !TEST_COMPILER_ICC
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
