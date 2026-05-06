//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__simd_>

// [simd.mask.unary], basic_mask unary operators
//
// constexpr basic_mask operator!() const noexcept;
// constexpr basic_vec<integer-from<Bytes>, Abi> operator+() const noexcept;
// constexpr basic_vec<integer-from<Bytes>, Abi> operator-() const noexcept;
// constexpr basic_vec<integer-from<Bytes>, Abi> operator~() const noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// operator!

template <int Bytes, int N>
TEST_FUNC constexpr void test_logical_not()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Mask mask(true);
  static_assert(cuda::std::is_same_v<decltype(!mask), Mask>);
  static_assert(noexcept(!mask));
  static_assert(is_const_member_function_v<decltype(&Mask::operator!)>);
  unused(mask);

  Mask all_true(true);
  Mask all_false(false);
  Mask mixed(is_even{});
  for (int i = 0; i < N; ++i)
  {
    assert((!all_true)[i] == false);
    assert((!all_false)[i] == true);
    assert((!mixed)[i] == (i % 2 != 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// operator+

template <int Bytes, int N>
TEST_FUNC constexpr void test_unary_plus()
{
  using Mask    = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  using Integer = integer_from_t<Bytes>;
  using Vec     = simd::basic_vec<Integer, simd::fixed_size<N>>;
  Mask mask(is_even{});
  static_assert(cuda::std::is_same_v<decltype(+mask), Vec>);
  static_assert(noexcept(+mask));

  auto vec = +mask;
  for (int i = 0; i < N; ++i)
  {
    if (i % 2 == 0)
    {
      assert(vec[i] == 1);
    }
    else
    {
      assert(vec[i] == 0);
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// operator-

template <int Bytes, int N>
TEST_FUNC constexpr void test_unary_minus()
{
  using Mask    = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  using Integer = integer_from_t<Bytes>;
  using Vec     = simd::basic_vec<Integer, simd::fixed_size<N>>;
  Mask mask(is_even{});
  static_assert(cuda::std::is_same_v<decltype(-mask), Vec>);
  static_assert(noexcept(-mask));

  Vec vec = -mask;
  for (int i = 0; i < N; ++i)
  {
    if (i % 2 == 0)
    {
      assert(vec[i] == static_cast<Integer>(-Integer{1}));
    }
    else
    {
      assert(vec[i] == Integer{0});
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// operator~

template <int Bytes, int N>
TEST_FUNC constexpr void test_bitwise_not()
{
  using Mask    = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  using Integer = integer_from_t<Bytes>;
  using Vec     = simd::basic_vec<Integer, simd::fixed_size<N>>;
  Mask mask(is_even{});
  static_assert(cuda::std::is_same_v<decltype(~mask), Vec>);
  static_assert(noexcept(~mask));

  Vec vec = ~mask;
  for (int i = 0; i < N; ++i)
  {
    if (i % 2 == 0)
    {
      assert(vec[i] == static_cast<Integer>(~Integer{1}));
    }
    else
    {
      assert(vec[i] == static_cast<Integer>(~Integer{0}));
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <int Bytes, int N>
TEST_FUNC constexpr void test_size()
{
  test_logical_not<Bytes, N>();
  test_unary_plus<Bytes, N>();
  test_unary_minus<Bytes, N>();
  test_bitwise_not<Bytes, N>();
}

template <int Bytes>
TEST_FUNC constexpr void test_bytes()
{
  test_size<Bytes, 1>();
  test_size<Bytes, 4>();
}

TEST_FUNC constexpr bool test()
{
  test_bytes<1>();
  test_bytes<2>();
  test_bytes<4>();
  test_bytes<8>();
#if _CCCL_HAS_INT128()
  test_bytes<16>();
#endif
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
