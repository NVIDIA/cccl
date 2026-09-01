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
// %PARAM% OP,FILECHECK_PREFIX_CALC op add=fetch_add,add_sub:sub=fetch_sub,add_sub:min=fetch_min,minmax:max=fetch_max,minmax
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,,non_seq_cst,no_acquire,no_membar:acquire=moa,,non_seq_cst,acquire,no_membar:release=more,ALL,non_seq_cst,no_acquire,membar:acq_rel=moar,ALL,non_seq_cst,acquire,membar:seq_cst=mosc,SC,seq_cst,acquire,membar
// %FILECHECK% PREFIX_COMBINE sm90,add_sub
// %FILECHECK% PREFIX_COMBINE sm100,add_sub
// %FILECHECK% PREFIX_COMBINE sm120,add_sub
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(cuda::atomic_ref<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

// clang-format off
/*

; SM90-PLUS-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.E.{{ADD|MIN|MAX}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; BLOCK: {{.*}}LD.E.128.STRONG.{{CTA|SM}} {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; NON_BLOCK: {{.*}}LD.E.128.STRONG.[[SASS_SCOPE]] {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; SM90_ADD_SUB: {{.*}}IADD3{{.*}}
; SM100_ADD_SUB: {{.*}}IADD3{{.*}}
; SM120_ADD_SUB-DAG: {{.*}}IADD.64 RZ, [[CARRY:P[0-9]+]], {{.*}}
; SM120_ADD_SUB-DAG: {{.*}}IADD.64 [[LOW:R[0-9]+]], {{.*}}
; SM120_ADD_SUB: {{.*}}IADD.64.X [[HIGH:R[0-9]+]], {{.*}}, [[CARRY]]{{.*}}
; MINMAX: {{.*}}ISETP{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{ADD|MIN|MAX}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; BLOCK: {{.*}}ATOM.E.CAS.128.STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, {{R[0-9]+}}, {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS.128.STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, {{R[0-9]+}}, {{R[0-9]+}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-DAG: {{.*}}{{ISETP\.NE(\.U32)?\.OR\.EX|ISETP\.NE\.U64\.OR|LOP3\.LUT}} [[RETRY_PRED:P[0-9]+]], {{.*}}[[RETRY_PRED]]{{.*}}
; NON_BLOCK_ACQUIRE-DAG: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{ADD|MIN|MAX}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX: {{.*}}@[[RETRY_PRED]] BRA{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{ADD|MIN|MAX}}{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
