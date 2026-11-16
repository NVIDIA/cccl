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

// [simd.reductions], scalar reduce, reduce_min, and reduce_max

#include <cuda/simd>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/concepts>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <class T>
TEST_HOST_DEVICE_FUNC constexpr void test_type()
{
  T value{7};
  static_assert(cuda::std::is_same_v<decltype(simd::reduce(value)), T>);
  static_assert(cuda::std::is_same_v<decltype(simd::reduce(value, true)), T>);
  static_assert(cuda::std::is_same_v<decltype(simd::reduce(value, false, cuda::std::multiplies<>{}, T{})), T>);
  static_assert(noexcept(simd::reduce(value)));
  static_assert(noexcept(simd::reduce(value, true)));
  static_assert(noexcept(simd::reduce(value, false, cuda::std::multiplies<>{}, T{})));

  assert(simd::reduce(value) == value);
  assert(simd::reduce(value, cuda::std::plus<>{}) == value);
  assert(simd::reduce(value, cuda::std::multiplies<>{}) == value);
  assert(simd::reduce(value, true) == value);
  assert(simd::reduce(value, false) == T{});
  assert(simd::reduce(value, true, cuda::std::multiplies<>{}) == value);
  assert(simd::reduce(value, false, cuda::std::multiplies<>{}) == T(1));
  assert(simd::reduce(value, true, cuda::std::plus<>{}, 0) == value);
  assert(simd::reduce(value, false, cuda::std::plus<>{}, 0) == T{});

  if constexpr (cuda::std::totally_ordered<T>)
  {
    static_assert(cuda::std::is_same_v<decltype(simd::reduce_min(value)), T>);
    static_assert(cuda::std::is_same_v<decltype(simd::reduce_max(value)), T>);
    static_assert(cuda::std::is_same_v<decltype(simd::reduce_min(value, true)), T>);
    static_assert(cuda::std::is_same_v<decltype(simd::reduce_max(value, true)), T>);
    static_assert(noexcept(simd::reduce_min(value)));
    static_assert(noexcept(simd::reduce_max(value)));
    static_assert(noexcept(simd::reduce_min(value, false)));
    static_assert(noexcept(simd::reduce_max(value, false)));

    assert(simd::reduce_min(value) == value);
    assert(simd::reduce_max(value) == value);
    assert(simd::reduce_min(value, true) == value);
    assert(simd::reduce_max(value, true) == value);
    assert(simd::reduce_min(value, false) == cuda::std::numeric_limits<T>::max());
    assert(simd::reduce_max(value, false) == cuda::std::numeric_limits<T>::lowest());
    if constexpr (cuda::std::numeric_limits<T>::has_infinity)
    {
      auto infinity = cuda::std::numeric_limits<T>::infinity();
      assert(simd::reduce(infinity) == infinity);
      assert(simd::reduce(infinity, true) == infinity);
      assert(simd::reduce_min(infinity, true) == infinity);
      assert(simd::reduce_max(-infinity, true) == -infinity);
    }
  }
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  test_type<signed char>();
  test_type<unsigned char>();
  test_type<char>();
  test_type<short>();
  test_type<unsigned short>();
  test_type<int>();
  test_type<unsigned int>();
  test_type<long>();
  test_type<unsigned long>();
  test_type<long long>();
  test_type<unsigned long long>();

  test_type<wchar_t>();
  test_type<char16_t>();
  test_type<char32_t>();
#if defined(__cccl_lib_char8_t)
  test_type<char8_t>();
#endif // defined(__cccl_lib_char8_t)

#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
  test_type<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  test_type<float>();
  test_type<double>();
  test_type<cuda::std::complex<float>>();
#if !_CCCL_OS(WINDOWS)
  // Windows lacks the int128 support required for 16-byte SIMD element representations.
  test_type<cuda::std::complex<double>>();
#endif // !_CCCL_OS(WINDOWS)
#if _CCCL_HAS_FLOAT128()
  test_type<__float128>();
#endif // _CCCL_HAS_FLOAT128()
  return true;
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_signed_zero()
{
  using Bits = integer_from_t<sizeof(T)>;
  T value(-0.0);
  auto negative_zero = cuda::std::bit_cast<Bits>(value);
  assert(cuda::std::bit_cast<Bits>(simd::reduce(value)) == negative_zero);
  assert(cuda::std::bit_cast<Bits>(simd::reduce(value, true)) == negative_zero);
  assert(cuda::std::bit_cast<Bits>(simd::reduce(value, true, cuda::std::plus<>{}, T{})) == negative_zero);
  assert(cuda::std::bit_cast<Bits>(simd::reduce_min(value)) == negative_zero);
  assert(cuda::std::bit_cast<Bits>(simd::reduce_max(value)) == negative_zero);
  assert(cuda::std::bit_cast<Bits>(simd::reduce_min(value, true)) == negative_zero);
  assert(cuda::std::bit_cast<Bits>(simd::reduce_max(value, true)) == negative_zero);
  assert(cuda::std::bit_cast<Bits>(simd::reduce(value, false)) == cuda::std::bit_cast<Bits>(T{}));
  assert(cuda::std::bit_cast<Bits>(simd::reduce(value, false, cuda::std::plus<>{}, value)) == negative_zero);
}

TEST_HOST_DEVICE_FUNC bool test_runtime()
{
  test_signed_zero<float>();
  test_signed_zero<double>();
#if _CCCL_HAS_FLOAT128()
  test_signed_zero<__float128>();
#endif // _CCCL_HAS_FLOAT128()

#if _LIBCUDACXX_HAS_NVFP16()
  test_type<__half>();
  test_signed_zero<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
  test_type<__nv_bfloat16>();
  test_signed_zero<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
  return true;
}

int main(int, char**)
{
  static_assert(test());
  // Also exercise the reductions at runtime.
  assert(test()); // NOLINT(misc-static-assert)
  assert(test_runtime());
  return 0;
}
