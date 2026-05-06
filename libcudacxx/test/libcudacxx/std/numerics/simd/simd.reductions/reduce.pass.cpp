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

// [simd.reductions], reduce
//
// template<class T, class Abi, class BinaryOperation = plus<>>
//   constexpr T reduce(const basic_vec<T, Abi>&, BinaryOperation = {});
//
// template<class T, class Abi, class BinaryOperation = plus<>>
//   constexpr T reduce(const basic_vec<T, Abi>&, const typename basic_vec<T, Abi>::mask_type&,
//                      BinaryOperation = {}, type_identity_t<T> identity_element = see below);

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../simd_test_utils.h"
#include "test_macros.h"

struct throwing_plus
{
  template <typename T1, typename T2>
  TEST_FUNC constexpr auto operator()(T1 a, T2 b) const
  {
    return a + b;
  }
};

//----------------------------------------------------------------------------------------------------------------------
// reduce with default plus<>

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_plus()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec   = make_iota_vec<T, N>();

  static_assert(cuda::std::is_same_v<decltype(simd::reduce(vec)), T>);
  static_assert(noexcept(simd::reduce(vec)));

  T result   = simd::reduce(vec);
  T expected = T{};
  for (int i = 0; i < N; ++i)
  {
    expected = static_cast<T>(expected + static_cast<T>(i));
  }
  assert(result == expected);
}

//----------------------------------------------------------------------------------------------------------------------
// reduce with explicit plus<>

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_explicit_plus()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(T{2});

  static_assert(noexcept(simd::reduce(vec, cuda::std::plus<>{})));

  T result = simd::reduce(vec, cuda::std::plus<>{});
  assert(result == static_cast<T>(T{2} * static_cast<T>(N)));
}

//----------------------------------------------------------------------------------------------------------------------
// reduce with multiplies<>

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_multiplies()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(T{2});

  static_assert(noexcept(simd::reduce(vec, cuda::std::multiplies<>{})));

  T result = simd::reduce(vec, cuda::std::multiplies<>{});
  T expected{1};
  for (int i = 0; i < N; ++i)
  {
    expected = static_cast<T>(expected * T{2});
  }
  assert(result == expected);
}

//----------------------------------------------------------------------------------------------------------------------
// reduce with mask and default plus<>

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_masked_plus()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec    = make_iota_vec<T, N>();
  Mask all_true(true);

  static_assert(cuda::std::is_same_v<decltype(simd::reduce(vec, all_true)), T>);
  static_assert(noexcept(simd::reduce(vec, all_true)));

  T result_all = simd::reduce(vec, all_true);
  T expected   = T{};
  for (int i = 0; i < N; ++i)
  {
    expected = static_cast<T>(expected + static_cast<T>(i));
  }
  assert(result_all == expected);

  Mask none_true(false);
  T result_none = simd::reduce(vec, none_true);
  assert(result_none == T{});
}

//----------------------------------------------------------------------------------------------------------------------
// reduce with mask and even-index elements

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_masked_even()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec    = make_iota_vec<T, N>();
  Mask even(is_even{});

  T result   = simd::reduce(vec, even);
  T expected = T{};
  for (int i = 0; i < N; ++i)
  {
    if (i % 2 == 0)
    {
      expected = static_cast<T>(expected + static_cast<T>(i));
    }
  }
  assert(result == expected);
}

//----------------------------------------------------------------------------------------------------------------------
// reduce with mask, explicit binary_op, and explicit identity_element

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_masked_explicit_identity()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec(T{3});
  Mask all_true(true);

  static_assert(noexcept(simd::reduce(vec, all_true, cuda::std::plus<>{}, cuda::std::declval<T>())));

  T result = simd::reduce(vec, all_true, cuda::std::plus<>{}, T{0});
  assert(result == static_cast<T>(T{3} * static_cast<T>(N)));

  Mask none_true(false);
  T result_none = simd::reduce(vec, none_true, cuda::std::plus<>{}, T{42});
  assert(result_none == T{42});
}

//----------------------------------------------------------------------------------------------------------------------
// reduce with mask and multiplies<>

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_masked_multiplies()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec(T{2});
  Mask all_true(true);

  static_assert(noexcept(simd::reduce(vec, all_true, cuda::std::multiplies<>{})));

  T result = simd::reduce(vec, all_true, cuda::std::multiplies<>{});
  T expected{1};
  for (int i = 0; i < N; ++i)
  {
    expected = static_cast<T>(expected * T{2});
  }
  assert(result == expected);

  Mask none_true(false);
  T result_none = simd::reduce(vec, none_true, cuda::std::multiplies<>{});
  assert(result_none == T{1});
}

//----------------------------------------------------------------------------------------------------------------------
// reduce with bit_and<>

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_bit_and()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec(static_cast<T>(0xFF));
  Mask none_true(false);

  static_assert(noexcept(simd::reduce(vec, cuda::std::bit_and<>{})));
  static_assert(noexcept(simd::reduce(vec, none_true, cuda::std::bit_and<>{})));

  T result = simd::reduce(vec, cuda::std::bit_and<>{});
  assert(result == static_cast<T>(0xFF));

  T result_none = simd::reduce(vec, none_true, cuda::std::bit_and<>{});
  assert(result_none == static_cast<T>(~T{}));
}

//----------------------------------------------------------------------------------------------------------------------
// reduce with bit_or<>

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_bit_or()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec(T{0});
  Mask none_true(false);

  static_assert(noexcept(simd::reduce(vec, cuda::std::bit_or<>{})));
  static_assert(noexcept(simd::reduce(vec, none_true, cuda::std::bit_or<>{})));

  T result = simd::reduce(vec, cuda::std::bit_or<>{});
  assert(result == T{0});

  T result_none = simd::reduce(vec, none_true, cuda::std::bit_or<>{});
  assert(result_none == T{});
}

//----------------------------------------------------------------------------------------------------------------------
// reduce with bit_xor<>

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_bit_xor()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec(T{1});
  Mask none_true(false);

  static_assert(noexcept(simd::reduce(vec, cuda::std::bit_xor<>{})));
  static_assert(noexcept(simd::reduce(vec, none_true, cuda::std::bit_xor<>{})));

  T result = simd::reduce(vec, cuda::std::bit_xor<>{});
  assert(result == static_cast<T>(N % 2 == 0 ? T{0} : T{1}));

  T result_none = simd::reduce(vec, none_true, cuda::std::bit_xor<>{});
  assert(result_none == T{});
}

//----------------------------------------------------------------------------------------------------------------------
// reduce noexcept with potentially-throwing operation

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_throwing_op()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec{};
  Mask mask(true);
  unused(vec, mask);

  static_assert(!noexcept(simd::reduce(vec, throwing_plus{})));
  static_assert(!noexcept(simd::reduce(vec, mask, throwing_plus{}, T{})));
}

//----------------------------------------------------------------------------------------------------------------------
// reduce with size 1

template <typename T>
TEST_FUNC constexpr void test_reduce_size_one()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<1>>;
  Vec vec(T{42});

  assert(simd::reduce(vec) == T{42});
  assert(simd::reduce(vec, cuda::std::multiplies<>{}) == T{42});
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_reduce_plus<T, N>();
  test_reduce_explicit_plus<T, N>();
  test_reduce_multiplies<T, N>();
  test_reduce_masked_plus<T, N>();
  test_reduce_masked_even<T, N>();
  test_reduce_masked_explicit_identity<T, N>();
  test_reduce_masked_multiplies<T, N>();
  test_reduce_throwing_op<T, N>();
  if constexpr (cuda::std::is_integral_v<T>)
  {
    test_reduce_bit_and<T, N>();
    test_reduce_bit_or<T, N>();
    test_reduce_bit_xor<T, N>();
  }
  if constexpr (N == 1)
  {
    test_reduce_size_one<T>();
  }
}

DEFINE_BASIC_VEC_TEST()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  static_assert(test());
  assert(test_runtime());
  return 0;
}
