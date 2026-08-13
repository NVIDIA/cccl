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
// %PARAM% TEMPLATE,SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE api vcad=vca,tsd,GPU,non_block:vcard=vcar,tsd,GPU,non_block:vcsa=vcsa,tss,SYS,non_block:vcsar=vcsar,tss,SYS,non_block
// %PARAM% TYPE,FILECHECK_PREFIX_TYPE type ptr1=char*,unscaled:ptr4=int32_t*,scaled
// %PARAM% OP op fetch_add:fetch_sub
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,,non_seq_cst,no_acquire,no_membar:acquire=moa,,non_seq_cst,acquire,no_membar:release=more,ALL,non_seq_cst,no_acquire,membar:acq_rel=moar,ALL,non_seq_cst,acquire,membar:seq_cst=mosc,SC,seq_cst,acquire,membar
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(TEMPLATE<TYPE, SCOPE>& atom, cuda::std::ptrdiff_t value)
{
  return atom.OP(value, ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.ADD.64{{.*}}
; UNSCALED-NOT: {{.*}}SHF.L.U64{{.*}}, {{R[0-9]+}}{{(\.reuse)?}}, 0x2{{.*}}
; SCALED: {{.*}}SHF.L.U64{{.*}}, {{R[0-9]+}}{{(\.reuse)?}}, 0x2{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.ADD.64{{.*}}
; BLOCK: {{.*}}ATOM.E.ADD.64.STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.ADD.64.STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; NON_BLOCK_ACQUIRE-NEXT: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.ADD.64{{.*}}
; UNSCALED: {{.*}}RET.ABS.NODEC{{.*}}
; SCALED: {{.*}}RET.ABS.NODEC{{.*}}

*/
