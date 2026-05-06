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

// [simd.reductions], reduce_max
//
// template<class T, class Abi>
//   constexpr T reduce_max(const basic_vec<T, Abi>&) noexcept;
//
// template<class T, class Abi>
//   constexpr T reduce_max(const basic_vec<T, Abi>&,
//                          const typename basic_vec<T, Abi>::mask_type&) noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// reduce_max without mask

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_max_basic()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec   = make_iota_vec<T, N>();

  static_assert(cuda::std::is_same_v<decltype(simd::reduce_max(vec)), T>);
  static_assert(noexcept(simd::reduce_max(vec)));

  T result = simd::reduce_max(vec);
  assert(result == static_cast<T>(N - 1));
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_max with all-equal elements

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_max_uniform()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(T{7});

  assert(simd::reduce_max(vec) == T{7});
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_max with maximum at first position

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_max_first()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(N - i);
  }
  Vec vec(arr);
  assert(simd::reduce_max(vec) == static_cast<T>(N));
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_max with mask: all true

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_max_masked_all()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec    = make_iota_vec<T, N>();
  Mask all_true(true);

  static_assert(cuda::std::is_same_v<decltype(simd::reduce_max(vec, all_true)), T>);
  static_assert(noexcept(simd::reduce_max(vec, all_true)));

  assert(simd::reduce_max(vec, all_true) == static_cast<T>(N - 1));
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_max with mask: none true returns numeric_limits<T>::lowest()

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_max_masked_none()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec(T{5});
  Mask none_true(false);

  assert(simd::reduce_max(vec, none_true) == cuda::std::numeric_limits<T>::lowest());
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_max with mask: even-index elements of iota

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_max_masked_even()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec    = make_iota_vec<T, N>();
  Mask even(is_even{});

  T result   = simd::reduce_max(vec, even);
  T expected = T{};
  for (int i = 0; i < N; ++i)
  {
    if (i % 2 == 0)
    {
      T val = static_cast<T>(i);
      if (val > expected)
      {
        expected = val;
      }
    }
  }
  assert(result == expected);
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_max with mask: single element selected

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_max_masked_single()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec    = make_iota_vec<T, N>();

  Mask first_only(is_index<0>{});
  assert(simd::reduce_max(vec, first_only) == T{0});
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_max with size 1

template <typename T>
TEST_FUNC constexpr void test_reduce_max_size_one()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<1>>;
  Vec vec(T{42});
  assert(simd::reduce_max(vec) == T{42});
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_reduce_max_basic<T, N>();
  test_reduce_max_uniform<T, N>();
  test_reduce_max_first<T, N>();
  test_reduce_max_masked_all<T, N>();
  test_reduce_max_masked_none<T, N>();
  test_reduce_max_masked_even<T, N>();
  test_reduce_max_masked_single<T, N>();
  if constexpr (N == 1)
  {
    test_reduce_max_size_one<T>();
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
