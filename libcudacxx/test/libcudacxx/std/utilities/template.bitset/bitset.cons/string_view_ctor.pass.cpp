//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//    template<class charT, class traits>
//        explicit bitset(
//            const basic_string_view<charT,traits>& str,
//            typename basic_string_view<charT,traits>::size_type pos = 0,
//            typename basic_string_view<charT,traits>::size_type n = basic_string_view<charT,traits>::npos,
//            charT zero = charT('0'), charT one = charT('1'));

#include <cuda/std/algorithm> // for 'min' and 'max'
#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/version>

#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
#  include <stdexcept> // for 'invalid_argument'
#endif // !TEST_COMPILER_NVRTC

#if TEST_HAS_EXCEPTIONS()
template <cuda::std::size_t N>
void test_exceptions()
{
  try
  {
    cuda::std::string_view s("xxx1010101010xxxx");
    cuda::std::bitset<N> v(s, s.size() + 1);
    assert(false);
  }
  catch (std::out_of_range&)
  {}
  try
  {
    cuda::std::string_view s("xxx1010101010xxxx");
    cuda::std::bitset<N> v(s, s.size() + 1, 10);
    assert(false);
  }
  catch (std::out_of_range&)
  {}
  try
  {
    cuda::std::string_view s("xxx1010101010xxxx");
    cuda::std::bitset<N> v(s);
    assert(false);
  }
  catch (std::invalid_argument&)
  {}
  try
  {
    cuda::std::string_view s("xxx1010101010xxxx");
    cuda::std::bitset<N> v(s, 2);
    assert(false);
  }
  catch (std::invalid_argument&)
  {}
  try
  {
    cuda::std::string_view s("xxx1010101010xxxx");
    cuda::std::bitset<N> v(s, 2, 10);
    assert(false);
  }
  catch (std::invalid_argument&)
  {}
  try
  {
    cuda::std::string_view s("xxxbababababaxxxx");
    cuda::std::bitset<N> v(s, 2, 10, 'a', 'b');
    assert(false);
  }
  catch (std::invalid_argument&)
  {}
}
void test_exceptions()
{
  test_exceptions<0>();
  test_exceptions<1>();
  test_exceptions<31>();
  test_exceptions<32>();
  test_exceptions<33>();
  test_exceptions<63>();
  test_exceptions<64>();
  test_exceptions<65>();
}
#endif // TEST_HAS_EXCEPTIONS()

template <cuda::std::size_t N>
__host__ __device__ constexpr void test_string_ctor()
{
  static_assert(!cuda::std::is_convertible_v<cuda::std::string_view, cuda::std::bitset<N>>, "");
  static_assert(cuda::std::is_constructible_v<cuda::std::bitset<N>, cuda::std::string_view>, "");
  {
    cuda::std::string_view s("1010101010");
    cuda::std::bitset<N> v(s);
    cuda::std::size_t M = cuda::std::min<cuda::std::size_t>(v.size(), 10);
    for (cuda::std::size_t i = 0; i < M; ++i)
    {
      assert(v[i] == (s[M - 1 - i] == '1'));
    }
    if constexpr (N > 10)
    {
      for (cuda::std::size_t i = 10; i < v.size(); ++i)
      {
        assert(v[i] == false);
      }
    }
  }
  {
    cuda::std::string_view s("xxx1010101010");
    cuda::std::bitset<N> v(s, 3);
    cuda::std::size_t M = cuda::std::min<cuda::std::size_t>(v.size(), 10);
    for (cuda::std::size_t i = 0; i < M; ++i)
    {
      assert(v[i] == (s[3 + M - 1 - i] == '1'));
    }
    if constexpr (N > 10)
    {
      for (cuda::std::size_t i = 10; i < v.size(); ++i)
      {
        assert(v[i] == false);
      }
    }
  }
  {
    cuda::std::string_view s("xxx1010101010xxxx");
    cuda::std::bitset<N> v(s, 3, 10);
    cuda::std::size_t M = cuda::std::min<cuda::std::size_t>(v.size(), 10);
    for (cuda::std::size_t i = 0; i < M; ++i)
    {
      assert(v[i] == (s[3 + M - 1 - i] == '1'));
    }
    if constexpr (N > 10)
    {
      for (cuda::std::size_t i = 10; i < v.size(); ++i)
      {
        assert(v[i] == false);
      }
    }
  }
  {
    cuda::std::string_view s("xxx1a1a1a1a1axxxx");
    cuda::std::bitset<N> v(s, 3, 10, 'a');
    cuda::std::size_t M = cuda::std::min<cuda::std::size_t>(v.size(), 10);
    for (cuda::std::size_t i = 0; i < M; ++i)
    {
      assert(v[i] == (s[3 + M - 1 - i] == '1'));
    }
    if constexpr (N > 10)
    {
      for (cuda::std::size_t i = 10; i < v.size(); ++i)
      {
        assert(v[i] == false);
      }
    }
  }
  {
    cuda::std::string_view s("xxxbababababaxxxx");
    cuda::std::bitset<N> v(s, 3, 10, 'a', 'b');
    cuda::std::size_t M = cuda::std::min<cuda::std::size_t>(v.size(), 10);
    for (cuda::std::size_t i = 0; i < M; ++i)
    {
      assert(v[i] == (s[3 + M - 1 - i] == 'b'));
    }
    if constexpr (N > 10)
    {
      for (cuda::std::size_t i = 10; i < v.size(); ++i)
      {
        assert(v[i] == false);
      }
    }
  }
}

struct Nonsense
{
  __host__ __device__ virtual ~Nonsense() {}
};

__host__ __device__ constexpr void test_for_non_eager_instantiation()
{
  // Ensure we don't accidentally instantiate `cuda::std::basic_string_view<Nonsense>`
  // since it may not be well formed and can cause an error in the
  // non-immediate context.
  static_assert(!cuda::std::is_constructible<cuda::std::bitset<3>, Nonsense*>::value, "");
  static_assert(
    !cuda::std::is_constructible<cuda::std::bitset<3>, Nonsense*, cuda::std::size_t, Nonsense&, Nonsense&>::value, "");
}

__host__ __device__ constexpr bool test()
{
  test_string_ctor<0>();
  test_string_ctor<1>();
  test_string_ctor<31>();
  test_string_ctor<32>();
  test_string_ctor<33>();
  test_string_ctor<63>();
  test_string_ctor<64>();
  test_string_ctor<65>();
  test_string_ctor<1000>();
  test_for_non_eager_instantiation();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
