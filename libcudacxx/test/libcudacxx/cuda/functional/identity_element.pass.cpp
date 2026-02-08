//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

#if _CCCL_COMPILER(GCC, >=, 12)
_CCCL_BEGIN_NV_DIAG_SUPPRESS(3215) // "if consteval" and "if not consteval" are not standard in this mode
_CCCL_DIAG_SUPPRESS_GCC("-Wc++23-extensions")
#endif // _CCCL_COMPILER(GCC, >=, 12)
#if _CCCL_CUDA_COMPILER(CLANG, >=, 13)
_CCCL_DIAG_SUPPRESS_CLANG("-Wc++23-extensions")
#endif // _CCCL_CUDA_COMPILER(CLANG, >=, 13)

/***********************************************************************************************************************
 * Helper
 **********************************************************************************************************************/

template <class T>
__host__ __device__ constexpr T get_value()
{
  if constexpr (cuda::std::is_same_v<T, bool>)
  {
    return true;
  }
  else
  {
    if constexpr (::cuda::std::__is_extended_floating_point_v<T>)
    {
      return cuda::std::__fp_max<T>();
    }
    else
    {
      return T{42};
    }
  }
}

/***********************************************************************************************************************
 * Identity Element
 **********************************************************************************************************************/

template <class Op, class T>
__host__ __device__ constexpr void test_identity_impl2()
{
  using U = cuda::std::remove_cv_t<T>;
  Op op{};
  auto value      = get_value<U>();
  auto identity1  = cuda::identity_element<Op, T>();
  auto result_lhs = static_cast<U>(op(value, identity1));
  auto result_rhs = static_cast<U>(op(identity1, value));
  assert(result_lhs == value);
  assert(result_rhs == value);
}

template <class Op, class T>
__host__ __device__ constexpr void test_identity_impl(bool has_identity, [[maybe_unused]] T identity)
{
  assert((has_identity == cuda::has_identity_element_v<Op, T>) );
  if constexpr (cuda::has_identity_element_v<Op, T>)
  {
    if constexpr (::cuda::is_floating_point_v<T>)
    {
#if _CCCL_COMPILER(GCC, >=, 8)
      assert(cuda::std::signbit(identity) == cuda::std::signbit(cuda::identity_element<Op, T>()));
#endif // _CCCL_COMPILER(GCC, >=, 8)
    }
    // handle extended floating-point types separately
    if constexpr (!::cuda::std::__is_extended_floating_point_v<T>)
    {
      assert((identity == cuda::identity_element<Op, T>()));
      test_identity_impl2<Op, T>();
      test_identity_impl2<Op, const T>();
      test_identity_impl2<Op, volatile T>();
      test_identity_impl2<Op, const volatile T>();
    }
#if _CCCL_CTK_AT_LEAST(12, 2) || _CCCL_DEVICE_COMPILATION()
    else
    {
      _CCCL_IF_NOT_CONSTEVAL_DEFAULT
      {
        assert((identity == cuda::identity_element<Op, T>()));
        test_identity_impl2<Op, T>();
        test_identity_impl2<Op, const T>();
      }
    }
#endif // _CCCL_CTK_AT_LEAST(12, 2) || _CCCL_DEVICE_COMPILATION()
  }
}

template <template <class...> class Op, class T>
__host__ __device__ constexpr void test_identity(bool has_identity, T identity)
{
  test_identity_impl<Op<T>, T>(has_identity, identity);
  test_identity_impl<Op<>, T>(has_identity, identity);
}

template <class T>
__host__ __device__ constexpr void test_identity_integral()
{
  test_identity<cuda::std::plus, T>(true, T{});
  test_identity<cuda::std::multiplies, T>(true, T{1});
  test_identity<cuda::std::bit_and, T>(true, static_cast<T>(~T{}));
  test_identity<cuda::std::bit_or, T>(true, T{});
  test_identity<cuda::std::bit_xor, T>(true, T{});
  test_identity<cuda::minimum, T>(true, cuda::std::numeric_limits<T>::max());
  test_identity<cuda::maximum, T>(true, cuda::std::numeric_limits<T>::lowest());
}

__host__ __device__ constexpr void test_identity_integral()
{
  test_identity<cuda::std::logical_and, bool>(true, true);
  test_identity<cuda::std::logical_or, bool>(true, false);
  test_identity_integral<signed char>();
  test_identity_integral<unsigned char>();
  test_identity_integral<short>();
  test_identity_integral<unsigned short>();
  test_identity_integral<int>();
  test_identity_integral<unsigned int>();
  test_identity_integral<long>();
  test_identity_integral<unsigned long>();
  test_identity_integral<long long>();
  test_identity_integral<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_identity_integral<__int128_t>();
  test_identity_integral<__uint128_t>();
#endif // _CCCL_HAS_INT128()
}

//----------------------------------------------------------------------------------------------------------------------
// floating-point

template <class T>
__host__ __device__ constexpr void test_identity_floating_point()
{
  test_identity<cuda::std::plus, T>(true, cuda::std::__fp_neg(T{}));
  test_identity<cuda::std::multiplies, T>(true, cuda::std::__fp_one<T>());
  test_identity<cuda::minimum, T>(true, cuda::std::numeric_limits<T>::infinity());
  test_identity<cuda::maximum, T>(true, cuda::std::__fp_neg(::cuda::std::__fp_inf<T>()));
}

__host__ __device__ constexpr void test_identity_floating_point()
{
  test_identity_floating_point<float>();
  test_identity_floating_point<double>();
#if _CCCL_HAS_FLOAT128()
  test_identity_floating_point<__float128>();
#endif // _CCCL_HAS_FLOAT128()
}

__host__ __device__ void test_identity_extended_floating_point()
{
#if _CCCL_HAS_NVFP16()
  test_identity_floating_point<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_identity_floating_point<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
}

/***********************************************************************************************************************
 * Negative tests - operators without identity elements
 *
 * - minus, divides, modulus: no identity (all types)
 **********************************************************************************************************************/

template <template <class...> class Op, class T>
__host__ __device__ constexpr bool no_identity()
{
  return !cuda::has_identity_element_v<Op<T>, T> && !cuda::has_identity_element_v<Op<>, T>;
}

template <class T>
__host__ __device__ constexpr void test_no_identity()
{
  static_assert(no_identity<cuda::std::minus, T>());
  static_assert(no_identity<cuda::std::divides, T>());
  static_assert(no_identity<cuda::std::modulus, T>());
}

__host__ __device__ constexpr void test_negative_integral()
{
  test_no_identity<signed char>();
  test_no_identity<unsigned char>();
  test_no_identity<short>();
  test_no_identity<unsigned short>();
  test_no_identity<int>();
  test_no_identity<unsigned int>();
  test_no_identity<long>();
  test_no_identity<unsigned long>();
  test_no_identity<long long>();
  test_no_identity<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_no_identity<__int128_t>();
  test_no_identity<__uint128_t>();
#endif // _CCCL_HAS_INT128()
}

__host__ __device__ constexpr void test_negative_floating_point()
{
  test_no_identity<float>();
  test_no_identity<double>();
#if _CCCL_HAS_FLOAT128()
  test_no_identity<__float128>();
#endif // _CCCL_HAS_FLOAT128()
}

__host__ __device__ void test_negative_extended_floating_point()
{
#if _CCCL_HAS_NVFP16()
  test_no_identity<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_no_identity<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
}

/***********************************************************************************************************************
 * Test dispatch
 **********************************************************************************************************************/

__host__ __device__ constexpr bool test()
{
  test_identity_integral();
  test_identity_floating_point();
  test_negative_integral();
  test_negative_floating_point();
  return true;
}

__host__ __device__ bool test_extended_floating_point()
{
  test_identity_extended_floating_point();
  test_negative_extended_floating_point();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  assert(test_extended_floating_point()); // run-time only
  return 0;
}
