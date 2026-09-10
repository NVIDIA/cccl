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
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,,non_seq_cst,no_acquire,no_membar:acquire=moa,,non_seq_cst,acquire,no_membar:release=more,ALL,non_seq_cst,no_acquire,membar:acq_rel=moar,ALL,non_seq_cst,acquire,membar:seq_cst=mosc,SC,seq_cst,acquire,membar
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(cuda::atomic_ref<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.exchange(value, ORDER);
}

/*

; SM90-PLUS-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.E.EXCH.STRONG{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH.STRONG{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; BLOCK: {{.*}}ATOM.E.EXCH.128.STRONG.{{CTA|SM}} PT, R4, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, R8{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.EXCH.128.STRONG.[[SASS_SCOPE]] PT, R4, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, R8{{.*}}
; NON_BLOCK_ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH.STRONG{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
