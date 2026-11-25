//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstdint>
#include <cuda/std/simd>

using s8x3 = cuda::std::simd::vec<cuda::std::int8_t, 3>;

__device__ s8x3 test_abs_s8x3(s8x3 x)
{
  return cuda::std::simd::abs(x);
}

/*
; SMXX-LABEL: {{.*}}Function : {{.*}}test_abs_s8x3{{.*}}
; NOT-SM100-PLUS-NOT: {{.*VABSDIFF4.*}}
; SM1XX: {{.*(VABSDIFF4 |VIADD\.S8x4 ).*}}
; SMXX: {{.*RET\..*}}
*/
