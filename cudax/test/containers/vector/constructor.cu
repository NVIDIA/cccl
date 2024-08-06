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

  SECTION("Construction with explicit size")
  {
    { // from resource, no alllocation
      const Vector vec{resource};
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // from resource and size, no alllocation
      const Vector vec{resource, 0};
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // from resource, size and value, no alllocation
      const Vector vec{resource, 0, T{42}};
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // from resource and size
      const Vector vec{resource, 5};
      CHECK(vec.capacity() == 5);
      CHECK(equal_range(vec, cuda::std::array<T, 5>{T(0), T(0), T(0), T(0), T(0)}));
    }

    { // from resource, size and value
      const Vector vec{resource, 5, T{42}};
      CHECK(vec.capacity() == 5);
      CHECK(equal_range(vec, cuda::std::array<T, 5>{T(42), T(42), T(42), T(42), T(42)}));
    }
  }

  SECTION("Construction from iterators")
  {
    const cuda::std::array<T, 4> input{T(1), T(42), T(1337), T(0)};
    { // can be constructed from two equal input iterators
      using iter = cpp17_input_iterator<const T*>;
      Vector vec(resource, iter{input.begin()}, iter{input.begin()});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // can be constructed from two equal forward iterators
      using iter = forward_iterator<const T*>;
      Vector vec(resource, iter{input.begin()}, iter{input.begin()});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

#if 0 // Implement growing
    { // can be constructed from two input iterators
      using iter = cpp17_input_iterator<const T*>;
      Vector vec(resource, iter{input.begin()}, iter{input.end()});
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, input));
    }
#endif // Implement growing

    { // can be constructed from two forward iterators
      using iter = forward_iterator<const T*>;
      Vector vec(resource, iter{input.begin()}, iter{input.end()});
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, input));
    }
  }

  SECTION("Construction from range")
  {
    { // can be constructed from an empty input range
      Vector vec(resource, input_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

#if 0 // Implement growing
    { // can be constructed from a non-empty input range
      Vector vec(resource, input_range<T, 4>{T(1), T(42), T(1337), T(0)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
    }
#endif // Implement growing

    { // can be constructed from an empty uncommon forward range
      Vector vec(resource, uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // can be constructed from a non-empty uncommon forward range
      Vector vec(resource, uncommon_range<T, 4>{T(1), T(42), T(1337), T(0)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
    }

    { // can be constructed from an empty sized uncommon forward range
      Vector vec(resource, sized_uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // can be constructed from a non-empty sized uncommon forward range
      Vector vec(resource, sized_uncommon_range<T, 4>{T(1), T(42), T(1337), T(0)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
    }

    { // can be constructed from an empty random access range
      Vector vec(resource, cuda::std::array<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // can be constructed from a non-empty random access range
      Vector vec(resource, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
    }
  }

  SECTION("Construction from initializer_list")
  {
    { // can be constructed from an empty initializer_list
      const cuda::std::initializer_list<T> input{};
      Vector vec(resource, input);
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // can be constructed from a non-empty initializer_list
      const cuda::std::initializer_list<T> input{T(1), T(42), T(1337), T(0)};
      Vector vec(resource, input);
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, input));
    }
  }
#if 0
  SECTION("copy construction")
  {
    static_assert(!cuda::std::is_nothrow_copy_constructible<Vector>::value, "");
    { // can be copy constructed from empty input
      const Vector input{resource, 0};
      Vector vec(input);
      CHECK(vec.empty());
    }

    { // can be copy constructed from non-empty input
      const Vector input{resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec(input);
      CHECK(!vec.empty());
      CHECK(equal_range(vec, input));
    }
  }

  SECTION("move construction")
  {
    static_assert(cuda::std::is_nothrow_move_constructible<Vector>::value, "");

    { // can be move constructed with empty input
      const Vector input{resource, 0};
      Vector vec(cuda::std::move(input));
      CHECK(vec.empty());
      CHECK(input.empty());
    }

    { // can be move constructed from non-empty input
      Vector input{resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec(cuda::std::move(input));
      CHECK(!vec.empty());
      CHECK(input.size() == 4);
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
    }
  }
#endif
}

#if 0

template <class T, template <class, size_t> class Range, class... Properties>
void test_range()
{
  using vector = cudax::vector<T, Properties...>;
  { // can be constructed from an empty range
    Vector vec(Range<T, 0>{});
    CHECK(vec.empty());
  }

  { // can be constructed from a non-empty range
    Vector vec(Range<T, 4>{T(1), T(42), T(1337), T(0)});
    CHECK(!vec.empty());
    CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
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
    CHECK(false);
  }

  try
  {
    vector too_small(2 * capacity, 42);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    CHECK(false);
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
    CHECK(false);
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
    CHECK(false);
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
    CHECK(false);
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
    CHECK(false);
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
    CHECK(false);
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
    CHECK(false);
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
    CHECK(false);
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
