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
// %PARAM% TEMPLATE,SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE api vcab=vca,tsb,CTA,block:vcad=vca,tsd,GPU,non_block:vcas=vca,tss,SYS,non_block:vcarb=vcar,tsb,CTA,block:vcard=vcar,tsd,GPU,non_block:vcars=vcar,tss,SYS,non_block:vcsa=vcsa,tss,SYS,non_block:vcsar=vcsar,tss,SYS,non_block
// %PARAM% OP,FILECHECK_PREFIX_OP op add=fetch_add,add:sub=fetch_sub,sub
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,,non_seq_cst,no_acquire,no_membar:acquire=moa,,non_seq_cst,acquire,no_membar:release=more,ALL,non_seq_cst,no_acquire,membar:acq_rel=moar,ALL,non_seq_cst,acquire,membar:seq_cst=mosc,SC,seq_cst,acquire,membar
// %FILECHECK% PREFIX_COMBINE add,block
// %FILECHECK% PREFIX_COMBINE add,non_block
// %FILECHECK% PREFIX_COMBINE sub,block
// %FILECHECK% PREFIX_COMBINE sub,non_block
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(TEMPLATE<int32_t, SCOPE>& atom, int32_t value)
{
  return atom.OP(value, ORDER);
}

// clang-format off
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.ADD{{(\.S32)?}}{{.*}}
; ADD-NOT: {{.*}}{{(IADD3|IADD|IMAD\.MOV)}} {{R[0-9]+}}, {{.*}}-{{R[0-9]+}}{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; SUB-DAG: {{.*}}{{(IADD3|IADD|IMAD\.MOV)}} [[OPERAND:R[0-9]+]], {{.*}}-{{R[0-9]+}}{{.*}}
; MEMBAR-DAG: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST-DAG: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.ADD{{(\.S32)?}}{{.*}}
; ADD_BLOCK: {{.*}}ATOM.E.ADD{{(\.S32)?}}.STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; ADD_NON_BLOCK: {{.*}}ATOM.E.ADD{{(\.S32)?}}.STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; SUB_BLOCK: {{.*}}ATOM.E.ADD{{(\.S32)?}}.STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, [[OPERAND]]{{.*}}
; SUB_NON_BLOCK: {{.*}}ATOM.E.ADD{{(\.S32)?}}.STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, [[OPERAND]]{{.*}}
; NON_BLOCK_ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.ADD{{(\.S32)?}}{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
