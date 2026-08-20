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
// %PARAM% TYPE,SASS_SIZE type i8=int8_t,.U8:u8=uint8_t,.U8:i16=int16_t,.U16:u16=uint16_t,.U16:f16=f16,.U16:bf16=bf16,.U16
// %PARAM% ORDER,FILECHECK_PREFIX_ORDER order relaxed=mor,non_sc:acquire=moa,non_sc:seq_cst=mosc,seq_cst
// clang-format on

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "atomic_codegen_helpers.h"

__device__ auto atomic_load(cuda::atomic_ref<volatile TYPE, SCOPE>& atom)
{
  return atom.load(ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_load.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*}}.STRONG{{.*}}
; SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; NON_SC-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*}}.STRONG{{.*}}
; BLOCK: {{.*}}LD.E[[SASS_SIZE]].STRONG.{{CTA|SM}} {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; NON_BLOCK: {{.*}}LD.E[[SASS_SIZE]].STRONG.[[SASS_SCOPE]] {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*}}.STRONG{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
