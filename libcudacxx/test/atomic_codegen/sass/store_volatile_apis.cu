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
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ORDER order relaxed=mor,,non_seq_cst,no_membar:release=more,ALL,non_seq_cst,membar:seq_cst=mosc,SC,seq_cst,membar
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ void atomic_codegen_test(TEMPLATE<int32_t, SCOPE>& atom, int32_t value)
{
  atom.store(value, ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX-NOT: {{.*}}ST.E{{.*}}.STRONG{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX-NOT: {{.*}}ST.E{{.*}}.STRONG{{.*}}
; BLOCK: {{.*}}ST.E.STRONG.{{CTA|SM}} {{.*}}, R6{{.*}}
; NON_BLOCK: {{.*}}ST.E.STRONG.[[SASS_SCOPE]] {{.*}}, R6{{.*}}
; SMXX-NOT: {{.*}}ST.E{{.*}}.STRONG{{.*}}
; SMXX-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NEXT: {{.*}}RET.ABS.NODEC{{.*}}

*/
