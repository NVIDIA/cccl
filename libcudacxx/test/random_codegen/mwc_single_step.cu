//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/random>
#include <cuda/std/utility>

// Return the updated engine and output to keep the complete step observable.
__device__ __noinline__ auto mwc64_cccl(cuda::experimental::mwc64x32 engine)
{
  const auto value = engine();
  return cuda::std::pair{engine, value};
}

__device__ __noinline__ auto mwc128_cccl(cuda::experimental::mwc128x32 engine)
{
  const auto value = engine();
  return cuda::std::pair{engine, value};
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*}}mwc128_cccl{{.*}}
; SMXX-NOT: {{.*}}{{IMAD.WIDE.U32|LOP3.LUT}}{{.*}}
; SMXX: {{.*}}IMAD.WIDE.U32{{.*}}-0x33339a,{{.*}}
; SMXX-NOT: {{.*}}{{IMAD.WIDE.U32|LOP3.LUT}}{{.*}}
; SMXX: {{.*}}LOP3.LUT{{.*}}0x3c, !PT{{.*}}
; SMXX-NOT: {{.*}}{{IMAD.WIDE.U32|LOP3.LUT}}{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*}}mwc64_cccl{{.*}}
; SMXX-NOT: {{.*}}{{IMAD.WIDE.U32|LOP3.LUT}}{{.*}}
; SMXX: {{.*}}IMAD.WIDE.U32{{.*}}-0xeba08,{{.*}}
; SMXX-NOT: {{.*}}{{IMAD.WIDE.U32|LOP3.LUT}}{{.*}}
; SMXX: {{.*}}LOP3.LUT{{.*}}0x3c, !PT{{.*}}
; SMXX-NOT: {{.*}}{{IMAD.WIDE.U32|LOP3.LUT}}{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
