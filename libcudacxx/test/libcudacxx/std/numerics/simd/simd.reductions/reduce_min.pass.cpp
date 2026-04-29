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

// [simd.reductions], reduce_min
//
// template<class T, class Abi>
//   constexpr T reduce_min(const basic_vec<T, Abi>&) noexcept;
//
// template<class T, class Abi>
//   constexpr T reduce_min(const basic_vec<T, Abi>&,
//                          const typename basic_vec<T, Abi>::mask_type&) noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// reduce_min without mask

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_min_basic()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec   = make_iota_vec<T, N>();

  static_assert(cuda::std::is_same_v<decltype(simd::reduce_min(vec)), T>);
  static_assert(noexcept(simd::reduce_min(vec)));

  T result = simd::reduce_min(vec);
  assert(result == T{0});
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_min with all-equal elements

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_min_uniform()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(T{7});

  assert(simd::reduce_min(vec) == T{7});
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_min with minimum at last position

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_min_last()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  cuda::std::array<T, N> arr{};
  for (int i = 0; i < N; ++i)
  {
    arr[i] = static_cast<T>(N - i);
  }
  Vec vec(arr);
  assert(simd::reduce_min(vec) == T{1});
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_min with mask: all true

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_min_masked_all()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec    = make_iota_vec<T, N>();
  Mask all_true(true);

  static_assert(cuda::std::is_same_v<decltype(simd::reduce_min(vec, all_true)), T>);
  static_assert(noexcept(simd::reduce_min(vec, all_true)));

  assert(simd::reduce_min(vec, all_true) == T{0});
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_min with mask: none true returns numeric_limits<T>::max()

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_min_masked_none()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec(T{5});
  Mask none_true(false);

  assert(simd::reduce_min(vec, none_true) == cuda::std::numeric_limits<T>::max());
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_min with mask: even-index elements

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_min_masked_even()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec    = make_iota_vec<T, N>();
  Mask even(is_even{});

  T result = simd::reduce_min(vec, even);
  assert(result == T{0});
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_min with mask: single element selected

template <typename T, int N>
TEST_FUNC constexpr void test_reduce_min_masked_single()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec    = make_iota_vec<T, N>();

  Mask last_only(is_index<N - 1>{});
  assert(simd::reduce_min(vec, last_only) == static_cast<T>(N - 1));
}

//----------------------------------------------------------------------------------------------------------------------
// reduce_min with size 1

template <typename T>
TEST_FUNC constexpr void test_reduce_min_size_one()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<1>>;
  Vec vec(T{42});
  assert(simd::reduce_min(vec) == T{42});
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_reduce_min_basic<T, N>();
  test_reduce_min_uniform<T, N>();
  test_reduce_min_last<T, N>();
  test_reduce_min_masked_all<T, N>();
  test_reduce_min_masked_none<T, N>();
  test_reduce_min_masked_even<T, N>();
  test_reduce_min_masked_single<T, N>();
  if constexpr (N == 1)
  {
    test_reduce_min_size_one<T>();
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
