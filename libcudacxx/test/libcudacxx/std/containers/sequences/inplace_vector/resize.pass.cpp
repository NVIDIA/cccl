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

struct is_one
{
  template <class T>
  __host__ __device__ constexpr bool operator()(const T& val) const noexcept
  {
    return val == T(1);
  }
};

template <class T>
__host__ __device__ constexpr void test()
{
  constexpr size_t max_capacity = 42ull;
  using vec                     = cuda::std::inplace_vector<T, max_capacity>;

  {
    cuda::std::initializer_list<T> expected{T(1)};
    vec resize_shrink{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    resize_shrink.resize(1);
    assert(cuda::std::equal(resize_shrink.begin(), resize_shrink.end(), expected.begin(), expected.end()));
  }

  {
    cuda::std::initializer_list<T> expected{T(1)};
    vec resize_value_shrink{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    resize_value_shrink.resize(1, T(5));
    assert(cuda::std::equal(resize_value_shrink.begin(), resize_value_shrink.end(), expected.begin(), expected.end()));
  }

  {
    cuda::std::initializer_list<T> expected{T(1), T(1337), T(42), T(12), T(0), T(-1), T(0), T(0), T(0)};
    vec resize_grow{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    resize_grow.resize(9);
    assert(cuda::std::equal(resize_grow.begin(), resize_grow.end(), expected.begin(), expected.end()));
  }

  {
    cuda::std::initializer_list<T> expected{T(1), T(1337), T(42), T(12), T(0), T(-1), T(5), T(5), T(5)};
    vec resize_value_grow{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    resize_value_grow.resize(9, T(5));
    assert(cuda::std::equal(resize_value_grow.begin(), resize_value_grow.end(), expected.begin(), expected.end()));
  }

  {
    vec clear{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    clear.clear();
    assert(clear.empty());
  }

  {
    cuda::std::initializer_list<T> expected{T(1), T(1337), T(42), T(12), T(0)};
    vec pop_back{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    pop_back.pop_back();
    assert(cuda::std::equal(pop_back.begin(), pop_back.end(), expected.begin(), expected.end()));
  }

  {
    cuda::std::initializer_list<T> expected_iter{T(1), T(42), T(12), T(0), T(-1)};
    vec erase_iter{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    auto res_erase_iter = erase_iter.erase(erase_iter.begin() + 1);
    static_assert(cuda::std::is_same<decltype(res_erase_iter), typename vec::iterator>::value, "");
    assert(*res_erase_iter == T(42));
    assert(cuda::std::equal(erase_iter.begin(), erase_iter.end(), expected_iter.begin(), expected_iter.end()));

    cuda::std::initializer_list<T> expected_iter_iter{T(1), T(12), T(0), T(-1)};
    vec erase_iter_iter{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    auto res_erase_iter_iter = erase_iter_iter.erase(erase_iter_iter.begin() + 1, erase_iter_iter.begin() + 3);
    static_assert(cuda::std::is_same<decltype(res_erase_iter_iter), typename vec::iterator>::value, "");
    assert(*res_erase_iter_iter == T(12));
    assert(cuda::std::equal(
      erase_iter_iter.begin(), erase_iter_iter.end(), expected_iter_iter.begin(), expected_iter_iter.end()));
  }

  {
    cuda::std::initializer_list<T> expected{T(1337), T(12), T(0), T(-1)};
    vec erased{T(1), T(1337), T(1), T(12), T(0), T(-1)};
    auto res_erase = erase(erased, T(1));
    static_assert(cuda::std::is_same<decltype(res_erase), typename vec::size_type>::value, "");
    assert(res_erase == 2);
    assert(cuda::std::equal(erased.begin(), erased.end(), expected.begin(), expected.end()));

    vec erased_if{T(1), T(1337), T(1), T(12), T(0), T(-1)};
    auto res_erase_if = erase_if(erased_if, is_one{});
    static_assert(cuda::std::is_same<decltype(res_erase_if), typename vec::size_type>::value, "");
    assert(res_erase_if == 2);
    assert(cuda::std::equal(erased_if.begin(), erased_if.end(), expected.begin(), expected.end()));
  }

  {
    cuda::std::initializer_list<T> expected{T(1), T(1337), T(1), T(12), T(0), T(-1)};
    vec shrink_to_fit{T(1), T(1337), T(1), T(12), T(0), T(-1)};
    shrink_to_fit.shrink_to_fit();
    assert(cuda::std::equal(shrink_to_fit.begin(), shrink_to_fit.end(), expected.begin(), expected.end()));
  }
  {
    cuda::std::initializer_list<T> expected{T(1), T(1337), T(1), T(12), T(0), T(-1)};
    vec reserve{T(1), T(1337), T(1), T(12), T(0), T(-1)};
    reserve.reserve(13);
    assert(cuda::std::equal(reserve.begin(), reserve.end(), expected.begin(), expected.end()));
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();

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
    using vec = cuda::std::inplace_vector<int, 42>;
    vec too_small{};
    try
    {
      too_small.resize(1337);
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
#if defined(_LIBCUDACXX_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // _LIBCUDACXX_IS_CONSTANT_EVALUATED

#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
