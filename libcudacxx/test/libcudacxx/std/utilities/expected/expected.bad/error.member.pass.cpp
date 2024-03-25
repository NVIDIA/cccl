//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: libcpp-no-exceptions

// E& error() & noexcept;
// const E& error() const & noexcept;
// E&& error() && noexcept;
// const E&& error() const && noexcept;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/utility>

template<class T, class = void>
constexpr bool WhatNoexcept = false;

template<class T>
constexpr bool WhatNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<T&&>().error())>> = noexcept(cuda::std::declval<T&&>().error());

static_assert(!ErrorNoexcept<int>);
static_assert(ErrorNoexcept<cuda::std::bad_expected_access<int>&>);
static_assert(ErrorNoexcept<cuda::std::bad_expected_access<int> const&>);
static_assert(ErrorNoexcept<cuda::std::bad_expected_access<int>&&>);
static_assert(ErrorNoexcept<cuda::std::bad_expected_access<int> const&&>);

__host__ __device__ void test() {
  // &
  {
    cuda::std::bad_expected_access<int> e(5);
    decltype(auto) i = e.error();
    static_assert(cuda::std::same_as<decltype(i), int&>);
    assert(i == 5);
  }

  // const &
  {
    const cuda::std::bad_expected_access<int> e(5);
    decltype(auto) i = e.error();
    static_assert(cuda::std::same_as<decltype(i), const int&>);
    assert(i == 5);
  }

  // &&
  {
    cuda::std::bad_expected_access<int> e(5);
    decltype(auto) i = cuda::std::move(e).error();
    static_assert(cuda::std::same_as<decltype(i), int&&>);
    assert(i == 5);
  }

  // const &&
  {
    const cuda::std::bad_expected_access<int> e(5);
    decltype(auto) i = cuda::std::move(e).error();
    static_assert(cuda::std::same_as<decltype(i), const int&&>);
    assert(i == 5);
  }
}

int main(int, char**) {
  test();
  return 0;
}
