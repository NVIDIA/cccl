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

// constexpr iterator_t<R> begin();
// constexpr sentinel_t<R> end();
// constexpr auto begin() const requires range<const R>;
// constexpr auto end() const requires range<const R>;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

struct Base
{
  __host__ __device__ constexpr int* begin()
  {
    return nullptr;
  }
  __host__ __device__ constexpr auto end()
  {
    return sentinel_wrapper<int*>(nullptr);
  }
  __host__ __device__ constexpr char* begin() const
  {
    return nullptr;
  }
  __host__ __device__ constexpr auto end() const
  {
    return sentinel_wrapper<char*>(nullptr);
  }
};
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<Base>, int*>);
static_assert(cuda::std::same_as<cuda::std::ranges::sentinel_t<Base>, sentinel_wrapper<int*>>);
static_assert(cuda::std::same_as<cuda::std::ranges::iterator_t<const Base>, char*>);
static_assert(cuda::std::same_as<cuda::std::ranges::sentinel_t<const Base>, sentinel_wrapper<char*>>);

struct NoConst
{
  __host__ __device__ int* begin();
  __host__ __device__ sentinel_wrapper<int*> end();
};

struct DecayChecker
{
  __host__ __device__ int*& begin() const;
  __host__ __device__ int*& end() const;
};

#if TEST_STD_VER >= 2020
template <class T>
concept HasBegin = requires(T t) { t.begin(); };

template <class T>
concept HasEnd = requires(T t) { t.end(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
constexpr bool HasBegin = false;

template <class T>
constexpr bool HasBegin<T, cuda::std::void_t<decltype(cuda::std::declval<T>().begin())>> = true;

template <class T, class = void>
constexpr bool HasEnd = false;

template <class T>
constexpr bool HasEnd<T, cuda::std::void_t<decltype(cuda::std::declval<T>().end())>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  {
    using OwningView = cuda::std::ranges::owning_view<Base>;
    OwningView ov;
    decltype(auto) b1 = static_cast<OwningView&>(ov).begin();
    decltype(auto) b2 = static_cast<OwningView&&>(ov).begin();
    decltype(auto) b3 = static_cast<const OwningView&>(ov).begin();
    decltype(auto) b4 = static_cast<const OwningView&&>(ov).begin();

    ASSERT_SAME_TYPE(decltype(b1), int*);
    ASSERT_SAME_TYPE(decltype(b2), int*);
    ASSERT_SAME_TYPE(decltype(b3), char*);
    ASSERT_SAME_TYPE(decltype(b4), char*);

    decltype(auto) e1 = static_cast<OwningView&>(ov).end();
    decltype(auto) e2 = static_cast<OwningView&&>(ov).end();
    decltype(auto) e3 = static_cast<const OwningView&>(ov).end();
    decltype(auto) e4 = static_cast<const OwningView&&>(ov).end();

    ASSERT_SAME_TYPE(decltype(e1), sentinel_wrapper<int*>);
    ASSERT_SAME_TYPE(decltype(e2), sentinel_wrapper<int*>);
    ASSERT_SAME_TYPE(decltype(e3), sentinel_wrapper<char*>);
    ASSERT_SAME_TYPE(decltype(e4), sentinel_wrapper<char*>);

    assert(b1 == e1);
    assert(b2 == e2);
    assert(b3 == e3);
    assert(b4 == e4);
  }
  {
    // NoConst has non-const begin() and end(); so does the owning_view.
    using OwningView = cuda::std::ranges::owning_view<NoConst>;
    static_assert(HasBegin<OwningView&>);
    static_assert(HasBegin<OwningView&&>);
    static_assert(!HasBegin<const OwningView&>);
    static_assert(!HasBegin<const OwningView&&>);
    static_assert(HasEnd<OwningView&>);
    static_assert(HasEnd<OwningView&&>);
    static_assert(!HasEnd<const OwningView&>);
    static_assert(!HasEnd<const OwningView&&>);
  }
  {
    // DecayChecker's begin() and end() return references; make sure the owning_view decays them.
    using OwningView = cuda::std::ranges::owning_view<DecayChecker>;
    OwningView ov;
    unused(ov);
    ASSERT_SAME_TYPE(decltype(ov.begin()), int*);
    ASSERT_SAME_TYPE(decltype(ov.end()), int*);
  }
  {
    // Test an empty view.
    int a[] = {1};
    auto ov = cuda::std::ranges::owning_view(cuda::std::ranges::subrange(a, a));
    assert(ov.begin() == a);
    assert(cuda::std::as_const(ov).begin() == a);
    assert(ov.end() == a);
    assert(cuda::std::as_const(ov).end() == a);
  }
  {
    // Test a non-empty view.
    int a[] = {1};
    auto ov = cuda::std::ranges::owning_view(cuda::std::ranges::subrange(a, a + 1));
    assert(ov.begin() == a);
    assert(cuda::std::as_const(ov).begin() == a);
    assert(ov.end() == a + 1);
    assert(cuda::std::as_const(ov).end() == a + 1);
  }
  {
    // Test a non-view.
    cuda::std::array<int, 2> a = {1, 2};
    auto ov                    = cuda::std::ranges::owning_view(cuda::std::move(a));
    assert(cuda::std::to_address(ov.begin()) != cuda::std::to_address(a.begin())); // because it points into the copy
    assert(cuda::std::to_address(cuda::std::as_const(ov).begin()) != cuda::std::to_address(a.begin()));
    assert(cuda::std::to_address(ov.end()) != cuda::std::to_address(a.end()));
    assert(cuda::std::to_address(cuda::std::as_const(ov).end()) != cuda::std::to_address(a.end()));
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
