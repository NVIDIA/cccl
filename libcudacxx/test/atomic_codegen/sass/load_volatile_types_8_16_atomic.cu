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
// %PARAM% SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE scope device=tsd,GPU,non_block
// %PARAM% TYPE type int8_t:uint8_t:int16_t:uint16_t:f16:bf16
// %PARAM% ORDER,FILECHECK_PREFIX_ORDER order relaxed=mor,non_sc:acquire=moa,non_sc:seq_cst=mosc,seq_cst
// clang-format on

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "atomic_codegen_helpers.h"

__device__ auto atomic_load(volatile cuda::atomic<TYPE, SCOPE>& atom)
{
  return atom.load(ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_load.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*}}.STRONG{{.*}}
; SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; NON_SC-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*}}.STRONG{{.*}}
; BLOCK: {{.*}}LD.E.STRONG.{{CTA|SM}} {{R[0-9]+}}, {{.*}}
; NON_BLOCK: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] {{R[0-9]+}}, {{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*}}.STRONG{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
