//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct Nothrow
{
  __host__ __device__ Nothrow() noexcept {}
};

struct NotDefaultable
{
  __host__ __device__ NotDefaultable() = delete;
  __host__ __device__ NotDefaultable(int) noexcept {}
};

struct MaybeThrowingDefault
{
  __host__ __device__ MaybeThrowingDefault() noexcept(false) {}
};

struct MaybeThrowingCopy
{
  __host__ __device__ MaybeThrowingCopy(const MaybeThrowingCopy&) noexcept(false) {}
};

struct MaybeThrowingMove
{
  __host__ __device__ MaybeThrowingMove(MaybeThrowingCopy&&) noexcept(false) {}
};

template <class Fn>
__host__ __device__ constexpr void test()
{
  using zip_function = cuda::zip_function<Fn>;
  static_assert(cuda::std::is_default_constructible_v<zip_function> == cuda::std::is_default_constructible_v<Fn>);
  static_assert(
    cuda::std::is_nothrow_default_constructible_v<zip_function> == cuda::std::is_nothrow_default_constructible_v<Fn>);

  static_assert(cuda::std::is_constructible_v<zip_function, Fn&&>);
  static_assert(cuda::std::is_constructible_v<zip_function, const Fn&>);
  static_assert(
    cuda::std::is_nothrow_constructible_v<zip_function, const Fn&> == cuda::std::is_nothrow_copy_constructible_v<Fn>);
  static_assert(
    cuda::std::is_nothrow_constructible_v<zip_function, Fn&&> == cuda::std::is_nothrow_move_constructible_v<Fn>);

  static_assert(cuda::std::is_copy_constructible_v<zip_function>);
  static_assert(cuda::std::is_move_constructible_v<zip_function>);
  static_assert(cuda::std::is_copy_assignable_v<zip_function>);
  static_assert(cuda::std::is_move_assignable_v<zip_function>);
}

__host__ __device__ constexpr void test()
{
  test<Nothrow>();
  test<NotDefaultable>();
  test<MaybeThrowingDefault>();
  test<MaybeThrowingCopy>();
  test<MaybeThrowingMove>();
}

int main(int, char**)
{
  return 0;
}
