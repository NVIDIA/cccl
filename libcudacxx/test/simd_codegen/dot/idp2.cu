//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// clang-format off
// %PARAM% TYPE_16,SASS_16_TYPE type16 signed=int16_t,S16:unsigned=uint16_t,U16
// %PARAM% TYPE_8,SASS_8_TYPE type8 signed=int8_t,S8:unsigned=uint8_t,U8
// %PARAM% LHS_TYPE,RHS_TYPE order forward=TYPE_16,TYPE_8:reverse=TYPE_8,TYPE_16
// clang-format on

#include <cuda/simd>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

namespace simd = cuda::std::simd;

using cuda::std::int16_t;
using cuda::std::int32_t;
using cuda::std::int8_t;
using cuda::std::uint16_t;
using cuda::std::uint32_t;
using cuda::std::uint8_t;

using lhs_vec_t = simd::basic_vec<LHS_TYPE, simd::fixed_size<4>>;
using rhs_vec_t = simd::basic_vec<RHS_TYPE, simd::fixed_size<4>>;
using accum_t =
  cuda::std::conditional_t<cuda::std::is_unsigned_v<TYPE_16> && cuda::std::is_unsigned_v<TYPE_8>, uint32_t, int32_t>;

extern "C" __device__ accum_t test_dot(lhs_vec_t lhs, rhs_vec_t rhs, accum_t init)
{
  return cuda::simd::dot(lhs, rhs, init);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : test_dot
; SMXX: {{.*IDP\.2A\.LO\.}}[[SASS_16_TYPE]]{{\.}}[[SASS_8_TYPE]]{{.*}}
; SMXX: {{.*IDP\.2A\.HI\.}}[[SASS_16_TYPE]]{{\.}}[[SASS_8_TYPE]]{{.*}}

*/
