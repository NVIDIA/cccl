//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>
#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/vector>

#include "types.h"
#include <catch2/catch.hpp>

// TODO: only device accessible resource
TEMPLATE_TEST_CASE("cudax::vector assignment",
                   "[container][vector]",
                   cuda::std::tuple<>,
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Resource      = typename extract_properties<TestType>::resource;
  using Resource_ref  = typename extract_properties<TestType>::resource_ref;
  using OtherResource = typename extract_properties<TestType>::other_resource;
  using Vector        = typename extract_properties<TestType>::vector;
  using T             = typename Vector::value_type;

  Resource raw_resource{};
  Resource_ref resource{raw_resource};

  SECTION("cudax::vector copy-assignment")
  {
    { // Can be copy-assigned an empty input
      const Vector input{resource};
      Vector vec{resource};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }
    { // Can be copy-assigned an empty input, shrinking
      const Vector input{resource};
      Vector vec{resource, 4};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // Can be copy-assigned a non-empty input, shrinking
      const Vector input{resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec{resource, {T(0), T(42), T(1337), T(42), T(5)}};
      vec = input;
      CHECK(!vec.empty());
      CHECK(equal_range(vec, input));
    }

    { // Can be copy-assigned an non-empty input growing from empty no reallocation
      const Vector input{resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec{resource, 42};
      vec.clear();
      vec = input;
      CHECK(vec.capacity() == 42);
      CHECK(equal_range(vec, input));
    }

    { // Can be copy-assigned an non-empty input growing from non-empty no reallocation
      const Vector input{resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec{resource, 42};
      vec.resize(2);
      vec = input;
      CHECK(vec.capacity() == 42);
      CHECK(equal_range(vec, input));
    }

#if 0 // Implement growing
    { // Can be copy-assigned a non-empty input, growing from empty with reallocation
      const Vector input{T(1), T(42), T(1337), T(0)};
      Vector vec{};
      vec = input;
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, input));
    }

    { // Can be copy-assigned a non-empty input, growing with reallocation
      const Vector input{T(1), T(42), T(1337), T(0)};
      Vector vec{resource, 42};
      vec = input;
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, input));
    }
#endif // Implement growing
  }

  SECTION("cudax::vector copy-assignment different resource")
  {
    OtherResource other_resource{resource};
    { // Can be copy-assigned an empty input
      const Vector input{other_resource};
      Vector vec{resource};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // Can be copy-assigned an empty input, shrinking
      const Vector input{other_resource};
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec = input;
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // Can be copy-assigned a non-empty input, shrinking
      const Vector input{other_resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec{resource, {T(0), T(42), T(1337), T(42), T(5)}};
      vec = input;
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, input));
    }

    { // Can be copy-assigned an non-empty input growing from empty without capacity
      const Vector input{other_resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec{resource};
      vec = input;
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, input));
    }

    { // Can be copy-assigned an non-empty input growing from empty with capacity
      const Vector input{other_resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec{resource, 42};
      vec.clear();
      vec = input;
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, input));
    }

    { // Can be copy-assigned an non-empty input growing from non-empty
      const Vector input{other_resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec{resource, {T(0), T(42)}};
      vec.resize(2);
      vec = input;
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, input));
    }
  }

  SECTION("cudax::vector move-assignment")
  {
    { // Can be move-assigned an empty input
      Vector input{resource};
      CHECK(input.empty());
      CHECK(input.data() == nullptr);

      Vector vec{resource};
      vec = cuda::std::move(input);
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an empty input, shrinking
      Vector input{resource};
      CHECK(input.empty());
      CHECK(input.data() == nullptr);

      Vector vec{resource, 4};
      vec = cuda::std::move(input);
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned a non-empty input, shrinking
      Vector input{resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec{resource, {T(0), T(42), T(1337), T(42), T(5)}};
      vec = cuda::std::move(input);
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an non-empty input growing from empty
      Vector input{resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec{resource};
      vec = cuda::std::move(input);
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an non-empty input growing from non-empty
      Vector input{resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec{resource, 42};
      vec.resize(2);
      vec = cuda::std::move(input);
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }
  }

  SECTION("cudax::vector copy-assignment different resource")
  {
    OtherResource other_resource{resource};
    { // Can be move-assigned an empty input
      Vector input{other_resource};
      Vector vec{resource};
      vec = cuda::std::move(input);
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }
    { // Can be move-assigned an empty input, shrinking
      Vector input{other_resource};
      Vector vec{resource, 4};
      vec = cuda::std::move(input);
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned a non-empty input, shrinking
      Vector input{other_resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec{resource, {T(0), T(42), T(1337), T(42), T(5)}};
      vec = cuda::std::move(input);
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an non-empty input growing from empty
      Vector input{other_resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec{resource};
      vec = cuda::std::move(input);
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }

    { // Can be move-assigned an non-empty input growing from non-empty
      Vector input{other_resource, {T(1), T(42), T(1337), T(0)}};
      Vector vec{resource, 42};
      vec.resize(2);
      vec = cuda::std::move(input);
      CHECK(vec.capacity() == 4);
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1), T(42), T(1337), T(0)}));
      CHECK(input.empty());
      CHECK(input.data() == nullptr);
    }
  }

  SECTION("cudax::vector assignment initializer_list")
  {
    const cuda::std::initializer_list<T> empty_input{};
    { // Can be assigned an empty initializer_list
      Vector vec{resource};
      vec = empty_input;
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // Can be assigned an empty initializer_list, shrinking
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      auto* old_ptr = vec.data();
      vec           = empty_input;
      CHECK(vec.empty());
      CHECK(vec.data() == old_ptr);
    }

    const cuda::std::initializer_list<T> input{T(1), T(42), T(1337), T(0)};
#if 0 // Implement growing
    { // Can be assigned a non-empty initializer_list, from empty
      Vector vec{resource};
      vec = input;
      CHECK(!vec.empty());
      CHECK(equal_range(vec, input));
    }
#endif // Implement growing

    { // Can be assigned a non-empty initializer_list, shrinking
      Vector vec{resource, {T(0), T(42), T(1337), T(42), T(5)}};
      vec = input;
      CHECK(!vec.empty());
      CHECK(vec.capacity() == 5);
      CHECK(equal_range(vec, input));
    }

#if 0 // Implement growing
    { // Can be assigned a non-empty initializer_list, growing from non empty
      Vector vec{resource, {T(0), T(42)}};
      vec = input;
      CHECK(!vec.empty());
      CHECK(equal_range(vec, input));
    }
#endif // Implement growing
  }
}

#if 0

#  ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{ // assignment throws std::bad_alloc
  constexpr size_t capacity = 4;
  using Vector              = cudax::vector<int, capacity>;
  Vector too_small{};

  try
  {
    cuda::std::initializer_list<int> input{0, 1, 2, 3, 4, 5, 6};
    too_small = input;
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
#  if defined(_LIBCUDACXX_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#  endif // _LIBCUDACXX_IS_CONSTANT_EVALUATED

#  ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#  endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
#endif
