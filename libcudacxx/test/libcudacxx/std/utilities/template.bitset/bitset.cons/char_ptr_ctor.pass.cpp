//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// template <class charT>
//     explicit bitset(const charT* str,
//                     typename basic_string_view<charT>::size_type n = basic_string_view<charT>::npos, //
//                     s/string/string_view since C++26 charT zero = charT('0'), charT one = charT('1')); // constexpr
//                     since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
// #include <cuda/std/algorithm> // for 'min' and 'max'
// #include <cuda/std/stdexcept> // for 'invalid_argument'

#include "test_macros.h"

// TEST_MSVC_DIAGNOSTIC_IGNORED(6294) // Ill-defined for-loop:  initial condition does not satisfy test.  Loop body not
// executed.
_CCCL_NV_DIAG_SUPPRESS(186)

#ifndef TEST_HAS_NO_EXCEPTIONS
template <cuda::std::size_t N>
void test_char_pointer_ctor_throw()
{
  try
  {
    cuda::std::bitset<N> v("xxx1010101010xxxx");
    assert(false);
  }
  catch (std::invalid_argument&)
  {}
}

void test_exceptions()
{
  test_char_pointer_ctor_throw<0>();
  test_char_pointer_ctor_throw<1>();
  test_char_pointer_ctor_throw<31>();
  test_char_pointer_ctor_throw<32>();
  test_char_pointer_ctor_throw<33>();
  test_char_pointer_ctor_throw<63>();
  test_char_pointer_ctor_throw<64>();
  test_char_pointer_ctor_throw<65>();
  test_char_pointer_ctor_throw<1000>();
}
#endif

template <cuda::std::size_t N>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_char_pointer_ctor()
{
  static_assert(!cuda::std::is_convertible<const char*, cuda::std::bitset<N>>::value, "");
  static_assert(cuda::std::is_constructible<cuda::std::bitset<N>, const char*>::value, "");
  {
    const char s[] = "1010101010";
    cuda::std::bitset<N> v(s);
    cuda::std::size_t M = cuda::std::min<cuda::std::size_t>(v.size(), 10);
    for (cuda::std::size_t i = 0; i < M; ++i)
    {
      assert(v[i] == (s[M - 1 - i] == '1'));
    }
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_ICC(186)
    for (cuda::std::size_t i = 10; i < v.size(); ++i)
    {
      _CCCL_DIAG_POP
      {
        assert(v[i] == false);
      }
    }
  }
  {
    const char s[] = "1010101010";
    cuda::std::bitset<N> v(s, 10);
    cuda::std::size_t M = cuda::std::min<cuda::std::size_t>(v.size(), 10);
    for (cuda::std::size_t i = 0; i < M; ++i)
    {
      assert(v[i] == (s[M - 1 - i] == '1'));
    }
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_ICC(186)
    for (cuda::std::size_t i = 10; i < v.size(); ++i)
    {
      _CCCL_DIAG_POP
      {
        assert(v[i] == false);
      }
    }
  }
  {
    const char s[] = "1a1a1a1a1a";
    cuda::std::bitset<N> v(s, 10, 'a');
    cuda::std::size_t M = cuda::std::min<cuda::std::size_t>(v.size(), 10);
    for (cuda::std::size_t i = 0; i < M; ++i)
    {
      assert(v[i] == (s[M - 1 - i] == '1'));
    }
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_ICC(186)
    for (cuda::std::size_t i = 10; i < v.size(); ++i)
    {
      _CCCL_DIAG_POP
      {
        assert(v[i] == false);
      }
    }
  }
  {
    const char s[] = "bababababa";
    cuda::std::bitset<N> v(s, 10, 'a', 'b');
    cuda::std::size_t M = cuda::std::min<cuda::std::size_t>(v.size(), 10);
    for (cuda::std::size_t i = 0; i < M; ++i)
    {
      assert(v[i] == (s[M - 1 - i] == 'b'));
    }
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_ICC(186)
    for (cuda::std::size_t i = 10; i < v.size(); ++i)
    {
      _CCCL_DIAG_POP
      {
        assert(v[i] == false);
      }
    }
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_char_pointer_ctor<0>();
  test_char_pointer_ctor<1>();
  test_char_pointer_ctor<31>();
  test_char_pointer_ctor<32>();
  test_char_pointer_ctor<33>();
  test_char_pointer_ctor<63>();
  test_char_pointer_ctor<64>();
  test_char_pointer_ctor<65>();
  test_char_pointer_ctor<1000>();

  return true;
}

int main(int, char**)
{
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif

  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
