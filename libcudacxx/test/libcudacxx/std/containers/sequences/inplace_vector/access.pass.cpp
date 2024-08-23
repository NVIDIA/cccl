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
__host__ __device__ constexpr void test()
{
  using vec = cuda::std::inplace_vector<T, 42>;
  vec range{T(1), T(1337), T(42), T(12), T(0), T(-1)};
  const vec const_range{T(0), T(42), T(1337), T(42), T(5), T(-42)};

  auto&& bracket = range[3];
  static_assert(cuda::std::is_same<decltype(bracket), typename vec::reference>::value, "");
  assert(bracket == T(12));

  range[3]              = T(4);
  auto&& bracket_assign = range[3];
  static_assert(cuda::std::is_same<decltype(bracket_assign), typename vec::reference>::value, "");
  assert(bracket_assign == T(4));

  auto&& const_bracket = const_range[3];
  static_assert(cuda::std::is_same<decltype(const_bracket), typename vec::const_reference>::value, "");
  assert(const_bracket == T(42));

  auto&& front = range.front();
  static_assert(cuda::std::is_same<decltype(front), typename vec::reference>::value, "");
  assert(front == T(1));

  auto&& const_front = const_range.front();
  static_assert(cuda::std::is_same<decltype(const_front), typename vec::const_reference>::value, "");
  assert(const_front == T(0));

  auto&& back = range.back();
  static_assert(cuda::std::is_same<decltype(back), typename vec::reference>::value, "");
  assert(back == T(-1));

  auto&& const_back = const_range.back();
  static_assert(cuda::std::is_same<decltype(const_back), typename vec::const_reference>::value, "");
  assert(const_back == -42);

  auto data = range.data();
  static_assert(cuda::std::is_same<decltype(data), typename vec::pointer>::value, "");
  assert(*data == T(1));
  assert(data == cuda::std::addressof(front));

  auto const_data = const_range.data();
  static_assert(cuda::std::is_same<decltype(const_data), typename vec::const_pointer>::value, "");
  assert(*const_data == T(0));
  assert(const_data == cuda::std::addressof(const_front));
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
{ // at throws std::out_of_range
  {
    using vec = cuda::std::inplace_vector<int, 42>;
    try
    {
      vec too_small{};
      auto res = too_small.at(5);
      unused(res);
    }
    catch (const std::out_of_range&)
    {}
    catch (...)
    {
      assert(false);
    }

    try
    {
      const vec too_small{};
      auto res = too_small.at(5);
      unused(res);
    }
    catch (const std::out_of_range&)
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
