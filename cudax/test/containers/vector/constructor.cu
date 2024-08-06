//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11

#include <cuda/memory_resource>
#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cuda/experimental/vector>

#include <stdexcept>

#include "types.h"
#include <catch2/catch.hpp>

template <class>
struct extract_properties;

template <class... Properties>
struct extract_properties<cuda::std::tuple<Properties...>>
{
  using vector   = cudax::vector<int, Properties...>;
  using resource = cuda::std::conditional_t<
    cudax::__select_execution_space<Properties...> == cudax::_ExecutionSpace::__host_device,
    cuda::mr::cuda_managed_memory_resource,
    cuda::std::conditional_t<cudax::__select_execution_space<Properties...> == cudax::_ExecutionSpace::__device,
                             cuda::mr::cuda_memory_resource,
                             host_memory_resource<int>>>;

  using resource_ref = cuda::mr::resource_ref<Properties...>;
};

TEMPLATE_TEST_CASE(
  "cudax::vector constructors", "[container][vector]", cuda::std::tuple<>, cuda::std::tuple<cuda::mr::host_accessible>)
{
  using Resource     = typename extract_properties<TestType>::resource;
  using Resource_ref = typename extract_properties<TestType>::resource_ref;
  using Vector       = typename extract_properties<TestType>::vector;
  using T            = typename Vector::value_type;

  Resource raw_resource{};
  Resource_ref resource{raw_resource};

  SECTION("Construction with zero size")
  {
    { // from resource
      const Vector vec{resource};
      assert(vec.empty());
    }

    { // from resource and size no allocation
      const Vector vec{resource, 0};
      assert(vec.empty());
    }

    { // from resource, size and value
      const Vector vec{resource, 0, T{42}};
      assert(vec.empty());
    }
  }

#if 0
  SECTION("copy construction")
  {
    static_assert(!cuda::std::is_nothrow_copy_constructible<Vector>::value, "");
    { // can be copy constructed from empty input
      const Vector input{resource, 0};
      Vector vec(input);
      assert(vec.empty());
    }

    { // can be copy constructed from non-empty input
      const Vector input{resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec(input);
      assert(!vec.empty());
      assert(equal_range(vec, input));
    }
  }

  SECTION("move construction")
  {
    static_assert(cuda::std::is_nothrow_move_constructible<Vector>::value, "");

    { // can be move constructed with empty input
      const Vector input{resource, 0};
      Vector vec(cuda::std::move(input));
      assert(vec.empty());
      assert(input.empty());
    }

    { // can be move constructed from non-empty input
      Vector input{resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec(cuda::std::move(input));
      assert(!vec.empty());
      assert(input.size() == 4);
      assert(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
    }
  }
#endif
}

#if 0
template <class T, class... Properties>
void test_size()
{
  using vector = cudax::vector<T, Properties...>;
  { // can be constructed from a size, is empty if zero
    vector vec(0);
    assert(vec.empty());
#  if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 10) && !defined(TEST_COMPILER_MSVC)
    static_assert(!noexcept(vector(0)), "");
#  endif // !TEST_COMPILER_GCC < 10 && !TEST_COMPILER_MSVC
  }

  { // can be constructed from a size, elements are value initialized
    constexpr size_t size{3};
    vector vec(size);
    assert(!vec.empty());
    assert(equal_range(vec, cuda::std::array<T, size>{T(0), T(0), T(0)}));
#  if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 10) && !defined(TEST_COMPILER_MSVC)
    static_assert(!noexcept(vector(3)), "");
#  endif // !TEST_COMPILER_GCC < 10 && !TEST_COMPILER_MSVC
  }
}

template <class T, class... Properties>
void test_size_value()
{
  using vector = cudax::vector<T, Properties...>;
  { // can be constructed from a size and a const T&, is empty if zero
    vector vec(0, T(42));
    assert(vec.empty());
#  if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 10) && !defined(TEST_COMPILER_MSVC)
    static_assert(!noexcept(vector(0, T(42))), "");
#  endif // !TEST_COMPILER_GCC < 10 && !TEST_COMPILER_MSVC
  }

  { // can be constructed from a size and a const T&, elements are copied
    constexpr size_t size{3};
    vector vec(size, T(42));
    assert(!vec.empty());
    assert(equal_range(vec, cuda::std::array<T, size>{T(42), T(42), T(42)}));
#  if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 10) && !defined(TEST_COMPILER_MSVC)
    static_assert(!noexcept(vector(3, T(42))), "");
#  endif // !TEST_COMPILER_GCC < 10 && !TEST_COMPILER_MSVC
  }
}

template <class T, class... Properties>
void test_iter()
{
  using vector = cudax::vector<T, Properties...>;
  { // can be constructed from two equal input iterators
    using iter = cpp17_input_iterator<const T*>;
    vector vec(iter{input.begin()}, iter{input.begin()});
    assert(vec.empty());
  }

  { // can be constructed from two equal forward iterators
    using iter = forward_iterator<const T*>;
    vector vec(iter{input.begin()}, iter{input.begin()});
    assert(vec.empty());
  }

  { // can be constructed from two input iterators
    using iter = cpp17_input_iterator<const T*>;
    vector vec(iter{input.begin()}, iter{input.end()});
    assert(!vec.empty());
    assert(equal_range(vec, input));
  }

  { // can be constructed from two forward iterators
    using iter = forward_iterator<const T*>;
    vector vec(iter{input.begin()}, iter{input.end()});
    assert(!vec.empty());
    assert(equal_range(vec, input));
  }
}

template <class T, class... Properties>
void test_init_list()
{
  using vector = cudax::vector<T, Properties...>;
  { // can be constructed from an empty initializer_list
    cuda::std::initializer_list<T> input{};
    vector vec(input);
    assert(vec.empty());
  }

  { // can be constructed from a non-empty initializer_list
    const cuda::std::initializer_list<T> input{T(1), T(42), T(1337), T(0)};
    vector vec(input);
    assert(!vec.empty());
    assert(equal_range(vec, input));
  }
}

template <class T, template <class, size_t> class Range, class... Properties>
void test_range()
{
  using vector = cudax::vector<T, Properties...>;
  { // can be constructed from an empty range
    vector vec(Range<T, 0>{});
    assert(vec.empty());
  }

  { // can be constructed from a non-empty range
    vector vec(Range<T, 4>{T(1), T(42), T(1337), T(0)});
    assert(!vec.empty());
    assert(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
  }
}

template <class T, class... Properties>
void test_range()
{
  test_range<T, input_range, Properties...>();
  test_range<T, uncommon_range, Properties...>();
  test_range<T, sized_uncommon_range, Properties...>();
  test_range<T, cuda::std::array, Properties...>();
}

template <class T, class... Properties>
void test()
{
  test_copy_move<T, Properties...>();
  test_size<T, Properties...>();
  test_size_value<T, Properties...>();
  test_iter<T, Properties...>();
  test_init_list<T, Properties...>();
  test_range<T, Properties...>();
}

template <class T>
void test()
{
  test<T>();
  test<T, cuda::mr::host_accessible>();
  test<T, cuda::mr::device_accessible>();
  test<T, cuda::mr::host_accessible, cuda::mr::device_accessible>();
  test<T, user_defined_property>();
}

__host__ __device__ bool test()
{
  test<int>();
  test<Trivial>();
  test<NonTrivial>();
  test<ThrowingDefaultConstruct>();
  test<ThrowingCopyConstructor>();
  test<ThrowingMoveConstructor>();
  test<ThrowingCopyAssignment>();
  test<ThrowingMoveAssignment>();
  test<NonTrivialDestructor>();

  return true;
}

#  ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{ // constructors throw std::bad_alloc
  using vector = cudax::vector<int>;

  try
  {
    vector too_small(2 * capacity);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    vector too_small(2 * capacity, 42);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    using iter = cpp17_input_iterator<const int*>;
    cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
    vector too_small(iter{input.begin()}, iter{input.end()});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
    vector too_small(input.begin(), input.end());
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    cuda::std::initializer_list<int> input{0, 1, 2, 3, 4, 5, 6};
    vector too_small(input);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    input_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
    vector too_small(input);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    uncommon_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
    vector too_small(input);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    sized_uncommon_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
    vector too_small(input);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }

  try
  {
    cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
    vector too_small(input);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    assert(false);
  }
}
#  endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();

#  ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#  endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
#endif
