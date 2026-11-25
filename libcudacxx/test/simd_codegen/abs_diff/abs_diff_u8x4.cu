//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/simd>
#include <cuda/std/cstdint>

namespace simd = cuda::std::simd;
using s8x4     = simd::vec<cuda::std::int8_t, 4>;

__device__ s8x4 test_abs_s8x4(s8x4 x)
{
  return simd::abs(x);
}

/*

; SMXX-LABEL: {{.*}}Function : {{.*}}test_abs_s8x4{{.*}}
; SM80: {{.*}}VABSDIFF4 {{R[0-9]+}}{{.*}}
; SM90: {{.*}}VABSDIFF4 {{R[0-9]+}}{{.*}}
; SM100: {{.*}}VABSDIFF4 {{R[0-9]+}}{{.*}}
;
; SM107-COUNT-2: {{.*}}VIMNMX.S8x4{{.*}}
; SM107: {{.*}}VIADD.S8x4{{.*}}
;
; SM120-COUNT-2: {{.*}}VIMNMX.S8x4{{.*}}
; SM120: {{.*}}VIADD.S8x4{{.*}}
;
; SM107f: {{.*}}VIADD.S8x4{{.*}}
; SM120f: {{.*}}VIADD.S8x4{{.*}}

*/
