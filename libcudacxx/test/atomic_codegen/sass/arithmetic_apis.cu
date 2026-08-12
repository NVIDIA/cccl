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
// %PARAM% TEMPLATE,SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE api cab=ca,tsb,CTA,block:cad=ca,tsd,GPU,non_block:cas=ca,tss,SYS,non_block:carb=car,tsb,CTA,block:card=car,tsd,GPU,non_block:cars=car,tss,SYS,non_block:csa=csa,tss,SYS,non_block:csar=csar,tss,SYS,non_block
// %PARAM% OP,FILECHECK_PREFIX_OP op add=fetch_add,add:sub=fetch_sub,sub
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_ORDER order relaxed=mor,,no_membar:acquire=moa,,no_membar:release=more,ALL,membar:acq_rel=moar,ALL,membar:seq_cst=mosc,SC,membar
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ auto atomic_arithmetic(TEMPLATE<int32_t, SCOPE>& atom, int32_t value)
{
  return atom.OP(value, ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_arithmetic.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; ADD-NOT: {{.*}}{{(IADD3|IMAD\.MOV)}} {{R[0-9]+}}, {{.*}}-R6{{.*}}
; SUB-DAG: {{.*}}{{(IADD3|IMAD\.MOV)}} [[OPERAND:R[0-9]+]], {{.*}}-R6{{.*}}
; MEMBAR-DAG: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; BLOCK: {{.*}}ATOM.E.ADD.S32.STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.ADD.S32.STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
