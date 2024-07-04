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
  constexpr size_t max_capacity         = 42ull;
  using vec                             = cuda::std::inplace_vector<T, max_capacity>;
  const cuda::std::array<T, 6> expected = {T(0), T(1), T(2), T(3), T(4), T(T(5))};

  {
    vec emplace             = {T(0), T(1), T(2), T(4), T(T(5))};
    const auto res_emplaced = emplace.emplace(emplace.begin() + 3, 3);
    static_assert(cuda::std::is_same<decltype(res_emplaced), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(emplace.begin(), emplace.end(), expected.begin(), expected.end()));
    assert(res_emplaced == emplace.begin() + 3);

    vec emplace_const            = {T(0), T(1), T(2), T(4), T(T(5))};
    const auto res_emplace_const = emplace_const.emplace(emplace_const.cbegin() + 3, 3);
    static_assert(cuda::std::is_same<decltype(res_emplace_const), const typename vec::iterator>::value, "");
    assert(cuda::std::equal(emplace_const.begin(), emplace_const.end(), expected.begin(), expected.end()));
    assert(res_emplace_const == emplace_const.cbegin() + 3);

    vec emplace_back        = {T(0), T(1), T(2), T(3), T(4)};
    auto&& res_emplace_back = emplace_back.emplace_back(5);
    static_assert(cuda::std::is_same<decltype(res_emplace_back), typename vec::reference>::value, "");
    assert(cuda::std::equal(emplace_back.begin(), emplace_back.end(), expected.begin(), expected.end()));
    assert(res_emplace_back == T(5));
    res_emplace_back = T(6);
    assert(res_emplace_back == T(6));

    vec push_back_lvalue        = {T(0), T(1), T(2), T(3), T(4)};
    const T to_be_pushed        = 5;
    auto&& res_push_back_lvalue = push_back_lvalue.push_back(to_be_pushed);
    static_assert(cuda::std::is_same<decltype(res_push_back_lvalue), typename vec::reference>::value, "");
    assert(cuda::std::equal(push_back_lvalue.begin(), push_back_lvalue.end(), expected.begin(), expected.end()));
    assert(res_push_back_lvalue == T(5));
    res_push_back_lvalue = T(6);
    assert(res_push_back_lvalue == T(6));

    vec push_back_rvalue        = {T(0), T(1), T(2), T(3), T(4)};
    auto&& res_push_back_rvalue = push_back_rvalue.push_back(T(5));
    static_assert(cuda::std::is_same<decltype(res_push_back_rvalue), typename vec::reference>::value, "");
    assert(cuda::std::equal(push_back_rvalue.begin(), push_back_rvalue.end(), expected.begin(), expected.end()));
    assert(res_push_back_rvalue == T(5));
    res_push_back_rvalue = T(6);
    assert(res_push_back_rvalue == T(6));
  }

  {
    vec emplace_back      = {T(0), T(1), T(2), T(3), T(4)};
    auto res_emplace_back = emplace_back.try_emplace_back(5);
    static_assert(cuda::std::is_same<decltype(res_emplace_back), typename vec::pointer>::value, "");
    assert(cuda::std::equal(emplace_back.begin(), emplace_back.end(), expected.begin(), expected.end()));
    assert(*res_emplace_back == T(5));
    *res_emplace_back = T(6);
    assert(*res_emplace_back == T(6));

    vec push_back_lvalue      = {T(0), T(1), T(2), T(3), T(4)};
    const T to_be_pushed      = 5;
    auto res_push_back_lvalue = push_back_lvalue.try_push_back(to_be_pushed);
    static_assert(cuda::std::is_same<decltype(res_push_back_lvalue), typename vec::pointer>::value, "");
    assert(cuda::std::equal(push_back_lvalue.begin(), push_back_lvalue.end(), expected.begin(), expected.end()));
    assert(*res_push_back_lvalue == T(5));
    *res_push_back_lvalue = T(6);
    assert(*res_push_back_lvalue == T(6));

    vec push_back_rvalue      = {T(0), T(1), T(2), T(3), T(4)};
    auto res_push_back_rvalue = push_back_rvalue.try_push_back(T(5));
    static_assert(cuda::std::is_same<decltype(res_push_back_rvalue), typename vec::pointer>::value, "");
    assert(cuda::std::equal(push_back_rvalue.begin(), push_back_rvalue.end(), expected.begin(), expected.end()));
    assert(*res_push_back_rvalue == T(5));
    *res_push_back_rvalue = T(6);
    assert(*res_push_back_rvalue == T(6));
  }

  {
    vec emplace_back        = {T(0), T(1), T(2), T(3), T(4)};
    auto&& res_emplace_back = emplace_back.unchecked_emplace_back(5);
    static_assert(cuda::std::is_same<decltype(res_emplace_back), typename vec::reference>::value, "");
    assert(cuda::std::equal(emplace_back.begin(), emplace_back.end(), expected.begin(), expected.end()));
    assert(res_emplace_back == T(5));
    res_emplace_back = T(6);
    assert(res_emplace_back == T(6));

    vec push_back_lvalue        = {T(0), T(1), T(2), T(3), T(4)};
    const T to_be_pushed        = 5;
    auto&& res_push_back_lvalue = push_back_lvalue.unchecked_push_back(to_be_pushed);
    static_assert(cuda::std::is_same<decltype(res_push_back_lvalue), typename vec::reference>::value, "");
    assert(cuda::std::equal(push_back_lvalue.begin(), push_back_lvalue.end(), expected.begin(), expected.end()));
    assert(res_push_back_lvalue == T(5));
    res_push_back_lvalue = T(6);
    assert(res_push_back_lvalue == T(6));

    vec push_back_rvalue        = {T(0), T(1), T(2), T(3), T(4)};
    auto&& res_push_back_rvalue = push_back_rvalue.unchecked_push_back(T(5));
    static_assert(cuda::std::is_same<decltype(res_push_back_rvalue), typename vec::reference>::value, "");
    assert(cuda::std::equal(push_back_rvalue.begin(), push_back_rvalue.end(), expected.begin(), expected.end()));
    assert(res_push_back_rvalue == T(5));
    res_push_back_rvalue = T(6);
    assert(res_push_back_rvalue == T(6));
  }

  // try_emplace and friends return nullptr when out of capacity
  using empty_vec = cuda::std::inplace_vector<T, 0>;
  {
    empty_vec emplace_back{};
    auto res_emplace_back = emplace_back.try_emplace_back(5);
    static_assert(cuda::std::is_same<decltype(res_emplace_back), typename vec::pointer>::value, "");
    assert(emplace_back.empty());
    assert(res_emplace_back == nullptr);

    empty_vec push_back_lvalue{};
    const T to_be_pushed      = 5;
    auto res_push_back_lvalue = push_back_lvalue.try_push_back(to_be_pushed);
    static_assert(cuda::std::is_same<decltype(res_push_back_lvalue), typename vec::pointer>::value, "");
    assert(push_back_lvalue.empty());
    assert(res_push_back_lvalue == nullptr);

    empty_vec push_back_rvalue{};
    auto res_push_back_rvalue = push_back_rvalue.try_push_back(T(5));
    static_assert(cuda::std::is_same<decltype(res_push_back_rvalue), typename vec::pointer>::value, "");
    assert(push_back_rvalue.empty());
    assert(res_push_back_rvalue == nullptr);
  }

  using small_vec = cuda::std::inplace_vector<T, 5>;
  {
    small_vec emplace_back = {T(0), T(1), T(2), T(3), T(4)};
    auto res_emplace_back  = emplace_back.try_emplace_back(5);
    static_assert(cuda::std::is_same<decltype(res_emplace_back), typename vec::pointer>::value, "");
    assert(cuda::std::equal(emplace_back.begin(), emplace_back.end(), expected.begin()));
    assert(res_emplace_back == nullptr);

    small_vec push_back_lvalue = {T(0), T(1), T(2), T(3), T(4)};
    const T to_be_pushed       = 5;
    auto res_push_back_lvalue  = push_back_lvalue.try_push_back(to_be_pushed);
    static_assert(cuda::std::is_same<decltype(res_push_back_lvalue), typename vec::pointer>::value, "");
    assert(cuda::std::equal(push_back_lvalue.begin(), push_back_lvalue.end(), expected.begin()));
    assert(res_push_back_lvalue == nullptr);

    small_vec push_back_rvalue = {T(0), T(1), T(2), T(3), T(4)};
    auto res_push_back_rvalue  = push_back_rvalue.try_push_back(T(5));
    static_assert(cuda::std::is_same<decltype(res_push_back_rvalue), typename vec::pointer>::value, "");
    assert(cuda::std::equal(push_back_rvalue.begin(), push_back_rvalue.end(), expected.begin()));
    assert(res_push_back_rvalue == nullptr);
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();

  if (!cuda::std::is_constant_evaluated())
  {
    test<NonTrivial>();
    test<NonTrivialDestructor>();
    test<ThrowingDefaultConstruct>();
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{ // emplace and friends throw std::bad_alloc when out of capacity
  using empty_vec = cuda::std::inplace_vector<int, 0>;
  {
    empty_vec empty{};
    try
    {
      auto emplace = empty.emplace_back(5);
      unused(emplace);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }

    try
    {
      const int input       = 5;
      auto push_back_lvalue = empty.push_back(input);
      unused(push_back_lvalue);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }

    try
    {
      auto push_back_rvalue = empty.push_back(5);
      unused(push_back_rvalue);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }
  }

  using small_vec = cuda::std::inplace_vector<int, 5>;
  {
    small_vec full{0, 1, 2, 3, 4};
    try
    {
      auto emplace = full.emplace_back(5);
      unused(emplace);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }

    try
    {
      const int input       = 5;
      auto push_back_lvalue = full.push_back(input);
      unused(push_back_lvalue);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      assert(false);
    }

    try
    {
      auto push_back_rvalue = full.push_back(5);
      unused(push_back_rvalue);
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
