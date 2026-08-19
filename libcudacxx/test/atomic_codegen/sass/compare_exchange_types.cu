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
// %PARAM% TYPE,SASS_SIZE type i32=int32_t,:u32=uint32_t,:i64=int64_t,.64:u64=uint64_t,.64:f32=float,:f64=double,.64:ptr1=char*,.64:ptr4=int32_t*,.64
// %PARAM% CAS cas compare_exchange_weak:compare_exchange_strong
// %PARAM% SUCCESS_ORDER,FAILURE_ORDER,SASS_MEMBAR,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order rr=mor,mor,,non_seq_cst,no_acquire,no_membar:ar=moa,mor,,non_seq_cst,acquire,no_membar:aa=moa,moa,,non_seq_cst,acquire,no_membar:er=more,mor,ALL,non_seq_cst,no_acquire,membar:br=moar,mor,ALL,non_seq_cst,acquire,membar:ba=moar,moa,ALL,non_seq_cst,acquire,membar:sr=mosc,mor,SC,seq_cst,acquire,membar:sa=mosc,moa,SC,seq_cst,acquire,membar:ss=mosc,mosc,SC,seq_cst,acquire,membar
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ bool atomic_codegen_test(TEMPLATE<TYPE, SCOPE>& atom, TYPE& expected, TYPE desired)
{
  return atom.CAS(expected, desired, SUCCESS_ORDER, FAILURE_ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; BLOCK: {{.*}}ATOM.E.CAS[[SASS_SIZE]].STRONG.{{CTA|SM}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS[[SASS_SIZE]].STRONG.[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
