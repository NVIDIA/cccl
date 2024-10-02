//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11

#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/inplace_vector>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
#  include <stdexcept>
#endif // !TEST_HAS_NO_EXCEPTIONS

template <class T>
__host__ __device__ constexpr void test_resize()
{
  constexpr size_t max_capacity = 42ull;
  using inplace_vector          = cuda::std::inplace_vector<T, max_capacity>;

  { // inplace_vector<T, 0>::resize with a size
    cuda::std::inplace_vector<T, 0> vec{};
    vec.resize(0);
    assert(vec.empty());
  }

  { // inplace_vector<T, 0>::resize with a size and value
    cuda::std::inplace_vector<T, 0> vec{};
    vec.resize(0, T{42});
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::resize with a size, shrinking
    inplace_vector vec{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    vec.resize(1);
    assert(equal_range(vec, cuda::std::array<T, 1>{T(1)}));
  }

  { // inplace_vector<T, N>::resize with a size and value, shrinking
    inplace_vector vec{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    vec.resize(1, T(5));
    assert(equal_range(vec, cuda::std::array<T, 1>{T(1)}));
  }

  { // inplace_vector<T, N>::resize with a size, growing, new elements are value initialized
    inplace_vector vec{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    vec.resize(8);
    assert(equal_range(vec, cuda::std::array<T, 8>{T(1), T(1337), T(42), T(12), T(0), T(-1), T(0), T(0)}));
  }

  { // inplace_vector<T, N>::resize with a size and value, growing, new elements are copied
    inplace_vector vec{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    vec.resize(8, T(5));
    assert(equal_range(vec, cuda::std::array<T, 8>{T(1), T(1337), T(42), T(12), T(0), T(-1), T(5), T(5)}));
  }
}

template <class T>
__host__ __device__ constexpr void test_clear()
{
  constexpr size_t max_capacity = 42ull;
  using inplace_vector          = cuda::std::inplace_vector<T, max_capacity>;

  { // inplace_vector<T, 0>::clear
    cuda::std::inplace_vector<T, 0> vec{};
    vec.clear();
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::clear, from empty
    inplace_vector vec{};
    vec.clear();
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::clear
    inplace_vector vec{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    vec.clear();
    assert(vec.empty());
  }
}

struct is_one
{
  template <class T>
  __host__ __device__ constexpr bool operator()(const T& val) const noexcept
  {
    return val == T(1);
  }
};

template <class T>
__host__ __device__ constexpr void test_pop_back()
{
  constexpr size_t max_capacity = 42ull;
  using inplace_vector          = cuda::std::inplace_vector<T, max_capacity>;

  { // inplace_vector<T, N>::pop_back
    inplace_vector vec{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    vec.pop_back();
    assert(equal_range(vec, cuda::std::array<T, 5>{T(1), T(1337), T(42), T(12), T(0)}));
  }
}

template <class T>
__host__ __device__ constexpr void test_erase()
{
  constexpr size_t max_capacity = 42ull;
  using inplace_vector          = cuda::std::inplace_vector<T, max_capacity>;

  { // inplace_vector<T, N>::erase(iter)
    inplace_vector vec{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    auto res = vec.erase(vec.begin() + 1);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::iterator>::value, "");
    assert(*res == T(42));
    assert(equal_range(vec, cuda::std::array<T, 5>{T(1), T(42), T(12), T(0), T(-1)}));
  }

  { // inplace_vector<T, N>::erase(iter, iter), iterators are equal
    inplace_vector vec{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    auto res = vec.erase(vec.begin() + 1, vec.begin() + 1);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::iterator>::value, "");
    assert(*res == T(1337));
    assert(equal_range(vec, cuda::std::array<T, 6>{T(1), T(1337), T(42), T(12), T(0), T(-1)}));
  }

  { // inplace_vector<T, N>::erase(iter, iter)
    inplace_vector vec{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    auto res = vec.erase(vec.begin() + 1, vec.begin() + 3);
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::iterator>::value, "");
    assert(*res == T(12));
    assert(equal_range(vec, cuda::std::array<T, 4>{T(1), T(12), T(0), T(-1)}));
  }

  { // erase(inplace_vector<T, 0>, value)
    cuda::std::inplace_vector<T, 0> vec{};
    auto res = erase(vec, T(1));
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::size_type>::value, "");
    assert(res == 0);
    assert(vec.empty());
  }

  { // erase_if(inplace_vector<T, 0>, pred)
    cuda::std::inplace_vector<T, 0> vec{};
    auto res = erase_if(vec, is_one{});
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::size_type>::value, "");
    assert(res == 0);
    assert(vec.empty());
  }

  { // erase(inplace_vector<T, N>, value)
    inplace_vector vec{T(1), T(1337), T(1), T(12), T(0), T(-1)};
    auto res = erase(vec, T(1));
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::size_type>::value, "");
    assert(res == 2);
    assert(equal_range(vec, cuda::std::array<T, 4>{T(1337), T(12), T(0), T(-1)}));
  }

  { // erase_if(inplace_vector<T, N>, pred)
    inplace_vector vec{T(1), T(1337), T(1), T(12), T(0), T(-1)};
    auto res = erase_if(vec, is_one{});
    static_assert(cuda::std::is_same<decltype(res), typename inplace_vector::size_type>::value, "");
    assert(res == 2);
    assert(equal_range(vec, cuda::std::array<T, 4>{T(1337), T(12), T(0), T(-1)}));
  }
}

template <class T>
__host__ __device__ constexpr void test_shrink_to_fit()
{
  constexpr size_t max_capacity = 42ull;
  using inplace_vector          = cuda::std::inplace_vector<T, max_capacity>;

  { // inplace_vector<T, 0>::shrink_to_fit
    cuda::std::inplace_vector<T, 0> vec{};
    vec.shrink_to_fit();
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::shrink_to_fit
    inplace_vector vec{T(1), T(1337), T(1), T(12), T(0), T(-1)};
    vec.shrink_to_fit();
    assert(equal_range(vec, cuda::std::array<T, 6>{T(1), T(1337), T(1), T(12), T(0), T(-1)}));
  }
}

template <class T>
__host__ __device__ constexpr void test_reserve()
{
  constexpr size_t max_capacity = 42ull;
  using inplace_vector          = cuda::std::inplace_vector<T, max_capacity>;

  { // inplace_vector<T, 0>::reserve
    cuda::std::inplace_vector<T, 0> vec{};
    vec.reserve(0);
    assert(vec.empty());
  }

  { // inplace_vector<T, N>::reserve
    inplace_vector vec{T(1), T(1337), T(1), T(12), T(0), T(-1)};
    vec.reserve(13);
    assert(equal_range(vec, cuda::std::array<T, 6>{T(1), T(1337), T(1), T(12), T(0), T(-1)}));
  }
}

template <class T>
__host__ __device__ constexpr void test()
{
  test_resize<T>();
  test_clear<T>();
  test_pop_back<T>();
  test_erase<T>();
  test_shrink_to_fit<T>();
  test_reserve<T>();
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<Trivial>();

  if (!cuda::std::__libcpp_is_constant_evaluated())
  {
    test<NonTrivial>();
    test<NonTrivialDestructor>();
    test<ThrowingDefaultConstruct>();
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{ // resize and reserve throw std::bad_alloc
  {
    using inplace_vector = cuda::std::inplace_vector<int, 42>;
    inplace_vector too_small{};
    try
    {
      too_small.resize(1337);
      assert(false);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }

    try
    {
      too_small.resize(1337, 5);
      assert(false);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }

    try
    {
      too_small.reserve(1337);
      assert(false);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
