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
// %PARAM% TEMPLATE,SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE api cab=ca,tsb,CTA,block:card=car,tsd,GPU,non_block
// %PARAM% TYPE,SASS_SIZE,SASS_CALC type f32=float,,FSETP:f64=double,.64,DSETP
// %PARAM% OP op fetch_min:fetch_max
// %PARAM% ORDER,FILECHECK_PREFIX_ORDER order relaxed=mor,no_membar:acquire=moa,no_membar:release=more,release:acq_rel=moar,release:seq_cst=mosc,seq_cst
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ auto atomic_minmax(TEMPLATE<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

// clang-format off
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_minmax.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; BLOCK-DAG: {{.*}}LD.E[[SASS_SIZE]].STRONG.{{CTA|SM}} [[EXPECTED:R[0-9]+]], {{.*\[}}[[ATOM_ADDR:R[0-9]+]]{{(\.64)?\].*}}
; NON_BLOCK-DAG: {{.*}}LD.E[[SASS_SIZE]].STRONG.[[SASS_SCOPE]] [[EXPECTED:R[0-9]+]], {{.*\[}}[[ATOM_ADDR:R[0-9]+]]{{(\.64)?\].*}}
; SMXX-DAG: {{.*}}[[SASS_CALC]]{{.*}}
; SEQ_CST-DAG: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; RELEASE-DAG: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; BLOCK: {{.*}}ATOM.E.CAS[[SASS_SIZE]].STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, [[OLD:R[0-9]+]], {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, [[EXPECTED]], {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS[[SASS_SIZE]].STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, [[OLD:R[0-9]+]], {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, [[EXPECTED]], {{R[0-9]+}}{{.*}}
; SMXX: {{.*}}ISETP.NE{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
