//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA Complex++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

// <cuda/std/__simd_>

// [simd.complex.math] free functions: real, imag, abs, arg, norm, conj, proj,
// exp, log, log10, sqrt, polar, pow

// The execution gets stuck in nvcc 12.0 with msvc.
// UNSUPPORTED: msvc && nvcc-12.0

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

namespace simd = cuda::std::simd;

// Meaningful inputs for complex math tests: four complex values spanning all quadrants with mixed magnitudes
// Diverse angles (in radians) covering positive/negative values and different quadrants
template <typename T>
struct polar_theta_generator
{
  template <typename I>
  TEST_FUNC constexpr T operator()(I i) const noexcept
  {
    switch (static_cast<int>(i) & 3)
    {
      case 0:
        return T(0.3);
      case 1:
        return T(-1.4);
      case 2:
        return T(1.1);
      default:
        return T(-0.7);
    }
  }
};

// Meaningful exponents for pow(base, expo)
template <typename T>
struct pow_exponent_generator
{
  template <typename I>
  TEST_FUNC constexpr cuda::std::complex<T> operator()(I i) const noexcept
  {
    switch (static_cast<int>(i) & 3)
    {
      case 0:
        return cuda::std::complex<T>(T(0.5), T(0.3));
      case 1:
        return cuda::std::complex<T>(T(1.0), T(-0.5));
      case 2:
        return cuda::std::complex<T>(T(-0.7), T(0.8));
      default:
        return cuda::std::complex<T>(T(-1.2), T(-0.4));
    }
  }
};

//----------------------------------------------------------------------------------------------------------------------
// real() / imag() free functions

template <typename T, int N>
TEST_FUNC constexpr void test_real_imag_free()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;

  ComplexVec vec(complex_diverse_generator<T>{});

  RealVec reals = simd::real(vec);
  RealVec imags = simd::imag(vec);

  static_assert(cuda::std::is_same_v<decltype(reals), RealVec>);
  static_assert(cuda::std::is_same_v<decltype(imags), RealVec>);
  static_assert(noexcept(simd::real(vec)));
  static_assert(noexcept(simd::imag(vec)));

  for (int i = 0; i < N; ++i)
  {
    assert(reals[i] == cuda::std::real(vec[i]));
    assert(imags[i] == cuda::std::imag(vec[i]));
  }

  // member .real() / .imag() getters must agree with free functions
  RealVec member_reals = vec.real();
  RealVec member_imags = vec.imag();
  for (int i = 0; i < N; ++i)
  {
    assert(member_reals[i] == reals[i]);
    assert(member_imags[i] == imags[i]);
  }

  // complex constructor from separate real/imag vectors
  ComplexVec vec2(reals, imags);
  for (int i = 0; i < N; ++i)
  {
    assert(vec2[i] == vec[i]);
  }

  // complex constructor with real-only (imag defaults to zero)
  ComplexVec vec3(reals);
  for (int i = 0; i < N; ++i)
  {
    assert(vec3[i] == Complex(cuda::std::real(vec[i]), T(0)));
  }

  // member .real(v) / .imag(v) setters
  ComplexVec vec4(Complex(T(0), T(0)));
  vec4.real(reals);
  vec4.imag(imags);
  for (int i = 0; i < N; ++i)
  {
    assert(vec4[i] == vec[i]);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// conj() / norm()

template <typename T, int N>
TEST_FUNC constexpr void test_conj_norm()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;

  ComplexVec vec(complex_generator<T, 3, 4>{});

  static_assert(cuda::std::is_same_v<decltype(simd::conj(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::norm(vec)), RealVec>);
  static_assert(!noexcept(simd::conj(vec)));
  static_assert(!noexcept(simd::norm(vec)));

  ComplexVec vec_conj = simd::conj(vec);
  RealVec vec_norm    = simd::norm(vec);
  for (int i = 0; i < N; ++i)
  {
    is_fp_close(vec_conj[i], cuda::std::conj(vec[i]));
    is_fp_close(vec_norm[i], cuda::std::norm(vec[i]));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// arg()

template <typename T, int N>
TEST_FUNC void test_arg()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;

  ComplexVec vec(complex_generator<T, 1, 2>{});

  static_assert(cuda::std::is_same_v<decltype(simd::arg(vec)), RealVec>);
  static_assert(!noexcept(simd::arg(vec)));

  RealVec vec_arg = simd::arg(vec);
  for (int i = 0; i < N; ++i)
  {
    is_fp_close(vec_arg[i], cuda::std::arg(vec[i]));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// abs()

template <typename T, int N>
TEST_FUNC void test_abs()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;

  ComplexVec vec(complex_generator<T, 3, 4>{});

  static_assert(cuda::std::is_same_v<decltype(simd::abs(vec)), RealVec>);
  static_assert(!noexcept(simd::abs(vec)));

  RealVec vec_abs = simd::abs(vec);
  for (int i = 0; i < N; ++i)
  {
    is_fp_close(vec_abs[i], cuda::std::abs(vec[i]));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// proj()

template <typename T, int N>
TEST_FUNC void test_proj()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(complex_generator<T, 3, 4>{});

  static_assert(cuda::std::is_same_v<decltype(simd::proj(vec)), ComplexVec>);
  static_assert(!noexcept(simd::proj(vec)));

  ComplexVec vec_proj = simd::proj(vec);
  for (int i = 0; i < N; ++i)
  {
    is_fp_close(vec_proj[i], cuda::std::proj(vec[i]));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// exp, log, log10

template <typename T, int N>
TEST_FUNC void test_exp_log()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(complex_generator<T, 1, 1>{});

  static_assert(cuda::std::is_same_v<decltype(simd::exp(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::log(vec)), ComplexVec>);
  static_assert(cuda::std::is_same_v<decltype(simd::log10(vec)), ComplexVec>);
  static_assert(!noexcept(simd::exp(vec)));
  static_assert(!noexcept(simd::log(vec)));
  static_assert(!noexcept(simd::log10(vec)));

  ComplexVec vec_exp   = simd::exp(vec);
  ComplexVec vec_log   = simd::log(vec);
  ComplexVec vec_log10 = simd::log10(vec);
  for (int i = 0; i < N; ++i)
  {
    is_fp_close(vec_exp[i], cuda::std::exp(vec[i]));
    is_fp_close(vec_log[i], cuda::std::log(vec[i]));
    is_fp_close(vec_log10[i], cuda::std::log10(vec[i]));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// sqrt()

template <typename T, int N>
TEST_FUNC void test_sqrt()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec vec(complex_generator<T, 4, 1>{});

  static_assert(cuda::std::is_same_v<decltype(simd::sqrt(vec)), ComplexVec>);
  static_assert(!noexcept(simd::sqrt(vec)));

  ComplexVec vec_sqrt = simd::sqrt(vec);
  for (int i = 0; i < N; ++i)
  {
    is_fp_close(vec_sqrt[i], cuda::std::sqrt(vec[i]));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// polar

template <typename T, int N>
TEST_FUNC void test_polar()
{
  using Complex    = cuda::std::complex<T>;
  using RealVec    = simd::basic_vec<T, simd::fixed_size<N>>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  RealVec rho(offset_generator<T, 5>{});
  RealVec theta(polar_theta_generator<T>{});

  static_assert(cuda::std::is_same_v<decltype(simd::polar(rho, theta)), ComplexVec>);
  static_assert(!noexcept(simd::polar(rho, theta)));

  ComplexVec result = simd::polar(rho, theta);
  for (int i = 0; i < N; ++i)
  {
    is_fp_close(result[i], cuda::std::polar(rho[i], theta[i]));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// pow

template <typename T, int N>
TEST_FUNC void test_pow()
{
  using Complex    = cuda::std::complex<T>;
  using ComplexVec = simd::basic_vec<Complex, simd::fixed_size<N>>;

  ComplexVec base(complex_diverse_generator<T>{});
  ComplexVec expo(pow_exponent_generator<T>{});

  static_assert(cuda::std::is_same_v<decltype(simd::pow(base, expo)), ComplexVec>);
  static_assert(!noexcept(simd::pow(base, expo)));

  ComplexVec result = simd::pow(base, expo);
  for (int i = 0; i < N; ++i)
  {
    // pow composes log + mul + exp; use the larger "composed" ULP budget.
    is_fp_close(result[i], cuda::std::pow(base[i], expo[i]));
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_constexpr()
{
  test_real_imag_free<T, N>();
  test_conj_norm<T, N>();
}

template <typename T, int N>
TEST_FUNC void test_runtime()
{
  test_arg<T, N>();
  test_abs<T, N>();
  test_proj<T, N>();
  test_exp_log<T, N>();
  test_sqrt<T, N>();
  test_polar<T, N>();
  test_pow<T, N>();
}

TEST_FUNC constexpr bool test()
{
  test_constexpr<float, 1>();
  test_constexpr<float, 4>();
#if _CCCL_HAS_INT128()
  test_constexpr<double, 1>();
  test_constexpr<double, 4>();
#endif // _CCCL_HAS_INT128()
  return true;
}

template <typename T, int N>
TEST_FUNC void test_type()
{
  test_constexpr<T, N>();
  test_runtime<T, N>();
}

TEST_FUNC bool test_runtime()
{
  test_runtime<float, 1>();
  test_runtime<float, 4>();
#if _CCCL_HAS_INT128()
  test_runtime<double, 1>();
  test_runtime<double, 4>();
#endif // _CCCL_HAS_INT128()
#if _LIBCUDACXX_HAS_NVFP16()
  test_type<__half, 1>();
  test_type<__half, 4>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_type<__nv_bfloat16, 1>();
  test_type<__nv_bfloat16, 4>();
#endif // _LIBCUDACXX_HAS_NVBF16()
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  assert(test_runtime());
  return 0;
}
