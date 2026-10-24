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

// [simd.reductions]: nonempty masked reductions operate only on selected values.

#include <cuda/simd>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/limits>

#include "../simd_test_utils.h"
#include "test_macros.h"

struct selected_lane
{
  int lane;

  template <class I>
  TEST_HOST_DEVICE_FUNC constexpr bool operator()(I i) const
  {
    return i == lane;
  }
};

template <class T>
struct lane_value
{
  int lane;
  T value;

  template <class I>
  TEST_HOST_DEVICE_FUNC constexpr T operator()(I i) const
  {
    return i == lane ? value : T{};
  }
};

template <class T, int N>
TEST_HOST_DEVICE_FUNC constexpr void test_type()
{
  using Vec     = simd::vec<T, N>;
  using Mask    = typename Vec::mask_type;
  using Bits    = integer_from_t<sizeof(T)>;
  auto infinity = cuda::std::numeric_limits<T>::infinity();
  Vec positive{infinity};
  Vec negative{-infinity};
  Vec zero{T(-0.0)};

  for (int lane : {0, N / 2, N - 1})
  {
    Mask singleton{selected_lane{lane}}; // a mask where only i == lane is true
    Vec positive_selected{lane_value<T>{lane, infinity}}; // a vector where only i == lane is <value>
    Vec negative_selected{lane_value<T>{lane, -infinity}};
    Vec zero_selected{lane_value<T>{lane, T(-0.0)}};

    assert(simd::reduce_min(positive_selected, singleton) == infinity);
    assert(simd::reduce_max(negative_selected, singleton) == -infinity);
#if _CCCL_HAS_CONSTEXPR_BIT_CAST()
    assert(cuda::std::bit_cast<Bits>(simd::reduce(zero_selected, singleton)) == cuda::std::bit_cast<Bits>(T(-0.0)));
    assert(cuda::std::bit_cast<Bits>(simd::reduce_min(zero_selected, singleton)) == cuda::std::bit_cast<Bits>(T(-0.0)));
    assert(cuda::std::bit_cast<Bits>(simd::reduce_max(zero_selected, singleton)) == cuda::std::bit_cast<Bits>(T(-0.0)));
#endif // _CCCL_HAS_CONSTEXPR_BIT_CAST()
  }
  assert(simd::reduce_min(positive, Mask{true}) == infinity);
  assert(simd::reduce_max(negative, Mask{true}) == -infinity);
  assert(simd::reduce_min(positive, Mask{false}) == cuda::std::numeric_limits<T>::max());
  assert(simd::reduce_max(negative, Mask{false}) == cuda::std::numeric_limits<T>::lowest());
  assert(simd::reduce(zero, Mask{false}) == T(0));
  assert(simd::reduce(positive, Mask{false}, cuda::std::multiplies<>{}) == T(1));
}

template <int N>
TEST_HOST_DEVICE_FUNC constexpr void test_size()
{
  test_type<float, N>();
  test_type<double, N>();
#if _CCCL_HAS_FLOAT128()
  test_type<__float128, N>();
#endif // _CCCL_HAS_FLOAT128()
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  test_size<1>();
  test_size<3>();
  test_size<4>();
  return true;
}

TEST_HOST_DEVICE_FUNC bool test_runtime()
{
  _SIMD_TEST_FP16(1)
  _SIMD_TEST_FP16(3)
  _SIMD_TEST_FP16(4)
  _SIMD_TEST_BF16(1)
  _SIMD_TEST_BF16(3)
  _SIMD_TEST_BF16(4)
  return true;
}

int main(int, char**)
{
  static_assert(test());
  assert(test()); // NOLINT(misc-static-assert)
  assert(test_runtime());
  return 0;
}
