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
// %PARAM% TEMPLATE,SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE api cab=ca,tsb,CTA,block:card=car,tsd,GPU,non_block:csa=csa,tss,SYS,non_block
// %PARAM% TYPE,FILECHECK_PREFIX_WIDTH type i32=int32_t,word:u32=uint32_t,word:i64=int64_t,dword:u64=uint64_t,dword
// %PARAM% OP op fetch_add:fetch_sub
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,,non_seq_cst,no_acquire,no_membar:acquire=moa,,non_seq_cst,acquire,no_membar:release=more,ALL,non_seq_cst,no_acquire,membar:acq_rel=moar,ALL,non_seq_cst,acquire,membar:seq_cst=mosc,SC,seq_cst,acquire,membar
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// %FILECHECK% PREFIX_COMBINE word,block
// %FILECHECK% PREFIX_COMBINE word,non_block
// %FILECHECK% PREFIX_COMBINE dword,block
// %FILECHECK% PREFIX_COMBINE dword,non_block
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(TEMPLATE<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

// clang-format off
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
; WORD_BLOCK: {{.*}}ATOM.E.ADD{{(\.S32)?}}.STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; WORD_NON_BLOCK: {{.*}}ATOM.E.ADD{{(\.S32)?}}.STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; DWORD_BLOCK: {{.*}}ATOM.E.ADD.64.STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; DWORD_NON_BLOCK: {{.*}}ATOM.E.ADD.64.STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; NON_BLOCK_ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
