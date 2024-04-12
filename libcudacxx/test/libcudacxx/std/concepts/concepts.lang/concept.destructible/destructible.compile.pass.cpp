//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class T>
// concept destructible = is_nothrow_destructible_v<T>;

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

struct Empty
{};

struct Defaulted
{
  ~Defaulted() = default;
};
struct Deleted
{
  ~Deleted() = delete;
};

struct Noexcept
{
  __host__ __device__ ~Noexcept() noexcept;
};
struct NoexceptTrue
{
  __host__ __device__ ~NoexceptTrue() noexcept(true);
};
struct NoexceptFalse
{
  __host__ __device__ ~NoexceptFalse() noexcept(false);
};

struct Protected
{
protected:
  ~Protected() = default;
};
struct Private
{
private:
  ~Private() = default;
};

template <class T>
struct NoexceptDependant
{
  __host__ __device__ ~NoexceptDependant() noexcept(cuda::std::is_same_v<T, int>);
};

template <class T>
__host__ __device__ void test()
{
  static_assert(cuda::std::destructible<T> == cuda::std::is_nothrow_destructible_v<T>, "");
}

__host__ __device__ void test()
{
  test<Empty>();

  test<Defaulted>();
  test<Deleted>();

  test<Noexcept>();
  test<NoexceptTrue>();
  test<NoexceptFalse>();

  test<Protected>();
  test<Private>();

  test<NoexceptDependant<int>>();
  test<NoexceptDependant<double>>();

  test<bool>();
  test<char>();
  test<int>();
  test<double>();
}

// Required for MSVC internal test runner compatibility.
int main(int, char**)
{
  return 0;
}
