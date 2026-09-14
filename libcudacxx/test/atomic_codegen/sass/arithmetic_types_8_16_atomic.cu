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
// %PARAM% TYPE,SASS_TYPE type i8=int8_t,.S32:u8=uint8_t,:i16=int16_t,.S32:u16=uint16_t,
// %PARAM% OP,SASS_OP,FILECHECK_PREFIX_OP op add=fetch_add,ADD,add_sub:sub=fetch_sub,ADD,add_sub:min=fetch_min,MIN,minmax:max=fetch_max,MAX,minmax
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,,non_seq_cst,no_acquire,no_membar:acquire=moa,,non_seq_cst,acquire,no_membar:release=more,ALL,non_seq_cst,no_acquire,membar:acq_rel=moar,ALL,non_seq_cst,acquire,membar:seq_cst=mosc,SC,seq_cst,acquire,membar
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// %FILECHECK% PREFIX_COMBINE add_sub,block
// %FILECHECK% PREFIX_COMBINE add_sub,non_block
// %FILECHECK% PREFIX_COMBINE minmax,block
// %FILECHECK% PREFIX_COMBINE minmax,non_block
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(cuda::atomic<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; ADD_SUB_BLOCK: {{.*}}ATOM.E.ADD{{(\.S32)?}}.STRONG.{{CTA|SM}}{{.*}}
; ADD_SUB_NON_BLOCK: {{.*}}ATOM.E.ADD{{(\.S32)?}}.STRONG.[[SASS_SCOPE]]{{.*}}
; MINMAX_BLOCK: {{.*}}ATOM.E.[[SASS_OP]][[SASS_TYPE]].STRONG.{{CTA|SM}}{{.*}}
; MINMAX_NON_BLOCK: {{.*}}ATOM.E.[[SASS_OP]][[SASS_TYPE]].STRONG.[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
