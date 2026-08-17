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
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ORDER order relaxed=mor,,non_seq_cst,no_membar:release=more,ALL,non_seq_cst,membar:seq_cst=mosc,SC,seq_cst,membar
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ void atomic_codegen_test(TEMPLATE<TYPE, SCOPE>& atom, TYPE value)
{
  atom.store(value, ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; BLOCK: {{.*}}ST.E[[SASS_SIZE]].STRONG.{{CTA|SM}} {{.*}}, {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ST.E[[SASS_SIZE]].STRONG.[[SASS_SCOPE]] {{.*}}, {{R[0-9]+}}{{.*}}
; SMXX-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NEXT: {{.*}}RET.ABS.NODEC{{.*}}

*/
