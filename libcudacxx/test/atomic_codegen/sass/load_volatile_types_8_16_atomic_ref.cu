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
// %PARAM% ORDER,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,no_acquire,non_sc:acquire=moa,acquire,non_sc:seq_cst=mosc,acquire,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// clang-format on

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(cuda::atomic_ref<volatile TYPE, SCOPE>& atom)
{
  return atom.load(ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*}}.STRONG{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SC-NOT: {{.*}}CCTL.IVALL{{.*}}
; SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SC-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SC-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*}}.STRONG{{.*}}
; BLOCK: {{.*}}LD.E[[SASS_SIZE]].STRONG.{{CTA|SM}} {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; NON_BLOCK: {{.*}}LD.E[[SASS_SIZE]].STRONG.[[SASS_SCOPE]] {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; NON_BLOCK_ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*}}.STRONG{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
