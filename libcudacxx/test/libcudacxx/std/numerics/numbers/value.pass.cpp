//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/numbers>

#include <cuda/std/cassert>
#include <cuda/std/numbers>

#include <test_macros.h>

_CCCL_NV_DIAG_SUPPRESS(1046) // Suppress "floating-point value cannot be represented exactly"

__host__ __device__ constexpr bool test()
{
  // default constants
  assert(cuda::std::numbers::e == 0x1.5bf0a8b145769p+1);
  assert(cuda::std::numbers::log2e == 0x1.71547652b82fep+0);
  assert(cuda::std::numbers::log10e == 0x1.bcb7b1526e50ep-2);
  assert(cuda::std::numbers::pi == 0x1.921fb54442d18p+1);
  assert(cuda::std::numbers::inv_pi == 0x1.45f306dc9c883p-2);
  assert(cuda::std::numbers::inv_sqrtpi == 0x1.20dd750429b6dp-1);
  assert(cuda::std::numbers::ln2 == 0x1.62e42fefa39efp-1);
  assert(cuda::std::numbers::ln10 == 0x1.26bb1bbb55516p+1);
  assert(cuda::std::numbers::sqrt2 == 0x1.6a09e667f3bcdp+0);
  assert(cuda::std::numbers::sqrt3 == 0x1.bb67ae8584caap+0);
  assert(cuda::std::numbers::inv_sqrt3 == 0x1.279a74590331cp-1);
  assert(cuda::std::numbers::egamma == 0x1.2788cfc6fb619p-1);
  assert(cuda::std::numbers::phi == 0x1.9e3779b97f4a8p+0);

  // float constants
  assert(cuda::std::numbers::e_v<float> == 0x1.5bf0a8p+1f);
  assert(cuda::std::numbers::log2e_v<float> == 0x1.715476p+0f);
  assert(cuda::std::numbers::log10e_v<float> == 0x1.bcb7b15p-2f);
  assert(cuda::std::numbers::pi_v<float> == 0x1.921fb54p+1f);
  assert(cuda::std::numbers::inv_pi_v<float> == 0x1.45f306p-2f);
  assert(cuda::std::numbers::inv_sqrtpi_v<float> == 0x1.20dd76p-1f);
  assert(cuda::std::numbers::ln2_v<float> == 0x1.62e42fp-1f);
  assert(cuda::std::numbers::ln10_v<float> == 0x1.26bb1bp+1f);
  assert(cuda::std::numbers::sqrt2_v<float> == 0x1.6a09e6p+0f);
  assert(cuda::std::numbers::sqrt3_v<float> == 0x1.bb67aep+0f);
  assert(cuda::std::numbers::inv_sqrt3_v<float> == 0x1.279a74p-1f);
  assert(cuda::std::numbers::egamma_v<float> == 0x1.2788cfp-1f);
  assert(cuda::std::numbers::phi_v<float> == 0x1.9e3779ap+0f);

  // double constants
  assert(cuda::std::numbers::e_v<double> == 0x1.5bf0a8b145769p+1);
  assert(cuda::std::numbers::log2e_v<double> == 0x1.71547652b82fep+0);
  assert(cuda::std::numbers::log10e_v<double> == 0x1.bcb7b1526e50ep-2);
  assert(cuda::std::numbers::pi_v<double> == 0x1.921fb54442d18p+1);
  assert(cuda::std::numbers::inv_pi_v<double> == 0x1.45f306dc9c883p-2);
  assert(cuda::std::numbers::inv_sqrtpi_v<double> == 0x1.20dd750429b6dp-1);
  assert(cuda::std::numbers::ln2_v<double> == 0x1.62e42fefa39efp-1);
  assert(cuda::std::numbers::ln10_v<double> == 0x1.26bb1bbb55516p+1);
  assert(cuda::std::numbers::sqrt2_v<double> == 0x1.6a09e667f3bcdp+0);
  assert(cuda::std::numbers::sqrt3_v<double> == 0x1.bb67ae8584caap+0);
  assert(cuda::std::numbers::inv_sqrt3_v<double> == 0x1.279a74590331cp-1);
  assert(cuda::std::numbers::egamma_v<double> == 0x1.2788cfc6fb619p-1);
  assert(cuda::std::numbers::phi_v<double> == 0x1.9e3779b97f4a8p+0);

#if _CCCL_HAS_LONG_DOUBLE()
  // long double constants
  assert(cuda::std::numbers::e_v<long double> == 0x1.5bf0a8b145769p+1l);
  assert(cuda::std::numbers::log2e_v<long double> == 0x1.71547652b82fep+0l);
  assert(cuda::std::numbers::log10e_v<long double> == 0x1.bcb7b1526e50ep-2l);
  assert(cuda::std::numbers::pi_v<long double> == 0x1.921fb54442d18p+1l);
  assert(cuda::std::numbers::inv_pi_v<long double> == 0x1.45f306dc9c883p-2l);
  assert(cuda::std::numbers::inv_sqrtpi_v<long double> == 0x1.20dd750429b6dp-1l);
  assert(cuda::std::numbers::ln2_v<long double> == 0x1.62e42fefa39efp-1l);
  assert(cuda::std::numbers::ln10_v<long double> == 0x1.26bb1bbb55516p+1l);
  assert(cuda::std::numbers::sqrt2_v<long double> == 0x1.6a09e667f3bcdp+0l);
  assert(cuda::std::numbers::sqrt3_v<long double> == 0x1.bb67ae8584caap+0l);
  assert(cuda::std::numbers::inv_sqrt3_v<long double> == 0x1.279a74590331cp-1l);
  assert(cuda::std::numbers::egamma_v<long double> == 0x1.2788cfc6fb619p-1l);
  assert(cuda::std::numbers::phi_v<long double> == 0x1.9e3779b97f4a8p+0l);
#endif // _CCCL_HAS_LONG_DOUBLE()

  return true;
}

// Extended floating point types are not comparable in constexpr context
__host__ __device__ void test_ext_fp()
{
#if _LIBCUDACXX_HAS_NVFP16()
  // __half constants
  assert(cuda::std::numbers::e_v<__half> == __half{2.71875});
  assert(cuda::std::numbers::log2e_v<__half> == __half{1.4423828125});
  assert(cuda::std::numbers::log10e_v<__half> == __half{0.434326171875});
  assert(cuda::std::numbers::pi_v<__half> == __half{3.140625});
  assert(cuda::std::numbers::inv_pi_v<__half> == __half{0.318359375});
  assert(cuda::std::numbers::inv_sqrtpi_v<__half> == __half{0.56396484375});
  assert(cuda::std::numbers::ln2_v<__half> == __half{0.693359375});
  assert(cuda::std::numbers::ln10_v<__half> == __half{2.302734375});
  assert(cuda::std::numbers::sqrt2_v<__half> == __half{1.4140625});
  assert(cuda::std::numbers::sqrt3_v<__half> == __half{1.732421875});
  assert(cuda::std::numbers::inv_sqrt3_v<__half> == __half{0.5771484375});
  assert(cuda::std::numbers::egamma_v<__half> == __half{0.5771484375});
  assert(cuda::std::numbers::phi_v<__half> == __half{1.6181640625});
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::numbers::e_v<__nv_bfloat16> == __nv_bfloat16{2.71875f});
  assert(cuda::std::numbers::log2e_v<__nv_bfloat16> == __nv_bfloat16{1.4453125f});
  assert(cuda::std::numbers::log10e_v<__nv_bfloat16> == __nv_bfloat16{0.43359375f});
  assert(cuda::std::numbers::pi_v<__nv_bfloat16> == __nv_bfloat16{3.140625f});
  assert(cuda::std::numbers::inv_pi_v<__nv_bfloat16> == __nv_bfloat16{0.318359375f});
  assert(cuda::std::numbers::inv_sqrtpi_v<__nv_bfloat16> == __nv_bfloat16{0.5625f});
  assert(cuda::std::numbers::ln2_v<__nv_bfloat16> == __nv_bfloat16{0.69140625f});
  assert(cuda::std::numbers::ln10_v<__nv_bfloat16> == __nv_bfloat16{2.296875f});
  assert(cuda::std::numbers::sqrt2_v<__nv_bfloat16> == __nv_bfloat16{1.4140625f});
  assert(cuda::std::numbers::sqrt3_v<__nv_bfloat16> == __nv_bfloat16{1.734375f});
  assert(cuda::std::numbers::inv_sqrt3_v<__nv_bfloat16> == __nv_bfloat16{0.578125f});
  assert(cuda::std::numbers::egamma_v<__nv_bfloat16> == __nv_bfloat16{0.578125f});
  assert(cuda::std::numbers::phi_v<__nv_bfloat16> == __nv_bfloat16{1.6171875f});
#endif // _LIBCUDACXX_HAS_NVBF16()
}

int main(int, char**)
{
  test();
  test_ext_fp();
  static_assert(test(), "");
  return 0;
}
