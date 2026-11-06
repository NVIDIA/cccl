//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// SIMD host/device functions are not supported in tile mode.

// [simd.subscr], [simd.mask.subscr]: SIMD indices select lanes through dynamic permute.

#include <cuda/simd>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <typename V, typename I, typename = void>
inline constexpr bool has_subscript = false;

template <typename V, typename I>
inline constexpr bool
  has_subscript<V, I, cuda::std::void_t<decltype(cuda::std::declval<const V&>()[cuda::std::declval<const I&>()])>> =
    true;

struct unrelated_type
{};

template <typename V>
TEST_HOST_DEVICE_FUNC constexpr void test_participation()
{
  static_assert(has_subscript<V, simd::vec<int, 3>>);
  static_assert(!has_subscript<V, simd::vec<float, 3>>);
  static_assert(!has_subscript<V, simd::mask<int, 3>>);
  static_assert(!has_subscript<V, simd::vec<int, 0>>);
  static_assert(!has_subscript<V, unrelated_type>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const V&>()[0]), typename V::value_type>);
  static_assert(noexcept(cuda::std::declval<const V&>()[0]));
}

template <typename Index, int N, typename V>
TEST_HOST_DEVICE_FUNC constexpr void test_indexed(const V& source, int stride, int offset)
{
  using Indices    = simd::vec<Index, N>;
  using Result     = simd::resize_t<N, V>;
  using Value      = typename V::value_type;
  using Comparable = cuda::std::conditional_t<cuda::is_floating_point_v<Value>, float, Value>;
  cuda::std::array<Index, N> lanes{};
  for (int i = 0; i < N; ++i)
  {
    lanes[i] = static_cast<Index>((i * stride + offset) % V::size());
  }
  const Indices indices(lanes);
  static_assert(cuda::std::is_same_v<decltype(source[indices]), Result>);

  const auto selected = source[indices];
  const auto permuted = simd::permute(source, indices);
  for (int i = 0; i < N; ++i)
  {
    assert(static_cast<Comparable>(selected[i]) == static_cast<Comparable>(source[lanes[i]]));
    assert(static_cast<Comparable>(selected[i]) == static_cast<Comparable>(permuted[i]));
  }
}

template <typename Index, int N, typename V>
TEST_HOST_DEVICE_FUNC constexpr void test_patterns(const V& source)
{
  test_indexed<Index, N>(source, 1, 0); // identity or repeated sequence
  test_indexed<Index, N>(source, V::size() - 1, V::size() - 1); // reverse
  test_indexed<Index, N>(source, 0, V::size() - 1); // broadcast the last lane
  test_indexed<Index, N>(source, 5, 1 % V::size()); // mixed order with repeated indices
}

template <typename T, int SourceN, int ResultN, typename Index = int>
TEST_HOST_DEVICE_FUNC constexpr void test_sizes()
{
  cuda::std::array<T, SourceN> input{};
  for (int i = 0; i < SourceN; ++i)
  {
    input[i] = static_cast<T>(i);
  }
  const simd::vec<T, SourceN> values(input, simd::flag_convert);
  const simd::mask<T, SourceN> mask(is_even{});

  test_patterns<Index, ResultN>(values);
  test_patterns<Index, ResultN>(mask);
}

template <typename T>
TEST_HOST_DEVICE_FUNC constexpr void test_type()
{
  test_sizes<T, 1, 1>();
  test_sizes<T, 1, 4>();
  test_sizes<T, 3, 4>();
  test_sizes<T, 4, 3>();
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  test_participation<simd::vec<int, 4>>();
  test_participation<simd::mask<int, 4>>();

  test_type<char>();
  test_type<short>();
  test_type<int>();
  test_type<unsigned>();
  test_type<long long>();
  test_type<float>();
  test_type<double>();
  test_type<cuda::std::complex<float>>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
  test_type<cuda::std::complex<double>>();
#endif // _CCCL_HAS_INT128()
  test_type<wchar_t>();
  test_type<char16_t>();
  test_type<char32_t>();
#if _CCCL_HAS_CHAR8_T()
  test_type<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  return true;
}

TEST_HOST_DEVICE_FUNC bool test_runtime()
{
#if _LIBCUDACXX_HAS_NVFP16()
  test_type<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_type<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
  return true;
}

int main(int, char**)
{
  assert(test()); // NOLINT(misc-static-assert)
  static_assert(test());
  assert(test_runtime()); // NOLINT(misc-static-assert)
  return 0;
}
