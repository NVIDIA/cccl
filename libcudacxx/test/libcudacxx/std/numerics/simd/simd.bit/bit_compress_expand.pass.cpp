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
// error: calling a host device function in tile mode

// [simd.bit], bit_compress and bit_expand

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/simd>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"

static_assert(__cccl_lib_simd_bitops >= 202607L);

struct bit_compress_function
{
  template <typename Vec, typename Mask>
  TEST_HOST_DEVICE_FUNC auto operator()(const Vec& v, const Mask& mask) const -> decltype(simd::bit_compress(v, mask));
};

struct bit_expand_function
{
  template <typename Vec, typename Mask>
  TEST_HOST_DEVICE_FUNC auto operator()(const Vec& v, const Mask& mask) const -> decltype(simd::bit_expand(v, mask));
};

template <typename Operation>
TEST_HOST_DEVICE_FUNC constexpr void test_constraints()
{
  using Vec      = simd::vec<unsigned, 3>;
  using IntVec   = simd::vec<int, 3>;
  using FloatVec = simd::vec<float, 3>;

  static_assert(cuda::std::is_invocable_v<Operation, Vec, Vec>);
  static_assert(cuda::std::is_invocable_v<Operation, Vec, unsigned>);
  static_assert(!cuda::std::is_invocable_v<Operation, Vec, simd::vec<unsigned, 4>>);
  static_assert(!cuda::std::is_invocable_v<Operation, Vec, IntVec>);
  static_assert(!cuda::std::is_invocable_v<Operation, IntVec, IntVec>);
  static_assert(!cuda::std::is_invocable_v<Operation, IntVec, int>);
  static_assert(!cuda::std::is_invocable_v<Operation, FloatVec, float>);
  static_assert(!cuda::std::is_invocable_v<Operation, simd::vec<unsigned, 0>, unsigned>);
  static_assert(!cuda::std::is_invocable_v<Operation, unsigned, unsigned>);
}

template <typename T>
struct mask_values
{
  template <typename Index>
  TEST_HOST_DEVICE_FUNC constexpr T operator()(Index) const
  {
    constexpr T masks[] = {T{0}, T{0b1010'0101}, cuda::std::numeric_limits<T>::max()};
    return masks[Index::value % 3];
  }
};

template <typename T, int N>
TEST_HOST_DEVICE_FUNC constexpr void test_type()
{
  using Vec = simd::vec<T, N>;
  const Vec values(bit_values<T>{});
  const Vec masks(mask_values<T>{});
  constexpr T scalar_mask = T{0b1010'0101};

  static_assert(cuda::std::is_same_v<decltype(simd::bit_compress(values, masks)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(simd::bit_compress(values, scalar_mask)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(simd::bit_expand(values, masks)), Vec>);
  static_assert(cuda::std::is_same_v<decltype(simd::bit_expand(values, scalar_mask)), Vec>);
  static_assert(noexcept(simd::bit_compress(values, masks)));
  static_assert(noexcept(simd::bit_compress(values, scalar_mask)));
  static_assert(noexcept(simd::bit_expand(values, masks)));
  static_assert(noexcept(simd::bit_expand(values, scalar_mask)));

  const Vec compressed        = simd::bit_compress(values, masks);
  const Vec compressed_scalar = simd::bit_compress(values, scalar_mask);
  const Vec expanded          = simd::bit_expand(values, masks);
  const Vec expanded_scalar   = simd::bit_expand(values, scalar_mask);

  for (int i = 0; i < N; ++i)
  {
    assert(compressed[i] == cuda::std::bit_compress(values[i], masks[i]));
    assert(compressed_scalar[i] == cuda::std::bit_compress(values[i], scalar_mask));
    assert(expanded[i] == cuda::std::bit_expand(values[i], masks[i]));
    assert(expanded_scalar[i] == cuda::std::bit_expand(values[i], scalar_mask));
  }
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  test_constraints<bit_compress_function>();
  test_constraints<bit_expand_function>();
  test_type<unsigned char, 3>();
  test_type<unsigned, 1>();
  test_type<unsigned, 3>();
  test_type<unsigned long long, 3>();
  return true;
}

int main(int, char**)
{
  assert(test()); // NOLINT(misc-static-assert)
  static_assert(test());
  return 0;
}
