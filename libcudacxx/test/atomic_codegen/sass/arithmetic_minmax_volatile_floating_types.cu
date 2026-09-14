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
// %PARAM% TEMPLATE,SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE api vcad=vca,tsd,GPU,non_block:vcard=vcar,tsd,GPU,non_block
// %PARAM% TYPE,SASS_SIZE,SASS_CALC type f32=float,,FSETP:f64=double,.64,DSETP
// %PARAM% OP op fetch_min:fetch_max
// %PARAM% ORDER,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,non_seq_cst,no_acquire,no_membar:acquire=moa,non_seq_cst,acquire,no_membar:release=more,non_seq_cst,no_acquire,release:acq_rel=moar,non_seq_cst,acquire,release:seq_cst=mosc,seq_cst,acquire,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(TEMPLATE<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

// clang-format off
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-DAG: {{.*}}LD.E[[SASS_SIZE]].STRONG.{{CTA|SM}} {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR:R[0-9]+]]{{(\.64)?\].*}}
; NON_BLOCK-DAG: {{.*}}LD.E[[SASS_SIZE]].STRONG.[[SASS_SCOPE]] {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR:R[0-9]+]]{{(\.64)?\].*}}
; SMXX-DAG: {{.*}}[[SASS_CALC]]{{.*}}
; SEQ_CST-DAG: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST-DAG: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; RELEASE-DAG: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS[[SASS_SIZE]]{{.*}}
; BLOCK: {{.*}}ATOM.E.CAS[[SASS_SIZE]].STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, [[OLD:R[0-9]+]], {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, {{R[0-9]+}}, {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS[[SASS_SIZE]].STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, [[OLD:R[0-9]+]], {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, {{R[0-9]+}}, {{R[0-9]+}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS[[SASS_SIZE]]{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-DAG: {{.*}}ISETP.NE{{.*}} [[CAS_RESULT_PRED:P[0-9]+]],{{.*}}
; NON_BLOCK_ACQUIRE-DAG: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS[[SASS_SIZE]]{{.*}}
; SMXX: {{.*}}@{{!?}}[[CAS_RESULT_PRED]] BRA{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
