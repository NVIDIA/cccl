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
// %PARAM% SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE scope block=tsb,CTA,block:device=tsd,GPU,non_block:system=tss,SYS,non_block
// %PARAM% TYPE type i128:u128
// %PARAM% ORDER,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,no_acquire,non_sc:acquire=moa,acquire,non_sc:seq_cst=mosc,acquire,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(cuda::atomic_ref<volatile TYPE, SCOPE>& atom)
{
  return atom.load(ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*\[}}[[ATOM_ADDR]]{{(\.64)?(\+0x[0-9a-f]+)?\].*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SC-NOT: {{.*}}CCTL.IVALL{{.*}}
; SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SC-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SC-NOT: {{.*}}MEMBAR.SC.{{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*\[}}[[ATOM_ADDR]]{{(\.64)?(\+0x[0-9a-f]+)?\].*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; BLOCK: {{.*}}LD.E.128.STRONG.{{CTA|SM}} R4, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; NON_BLOCK: {{.*}}LD.E.128.STRONG.[[SASS_SCOPE]] R4, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; NON_BLOCK_ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*\[}}[[ATOM_ADDR]]{{(\.64)?(\+0x[0-9a-f]+)?\].*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
