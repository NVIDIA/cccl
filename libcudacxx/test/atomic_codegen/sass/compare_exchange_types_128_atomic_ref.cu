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
// %PARAM% CAS cas compare_exchange_weak:compare_exchange_strong
// %PARAM% SUCCESS_ORDER,FAILURE_ORDER,SASS_MEMBAR,FILECHECK_PREFIX_ORDER order rr=mor,mor,,no_membar:ar=moa,mor,,no_membar:aa=moa,moa,,no_membar:er=more,mor,ALL,membar:br=moar,mor,ALL,membar:ba=moar,moa,ALL,membar:sr=mosc,mor,SC,membar:sa=mosc,moa,SC,membar:ss=mosc,mosc,SC,membar
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ bool atomic_compare_exchange(cuda::atomic_ref<TYPE, SCOPE>& atom, TYPE& expected, TYPE desired)
{
  return atom.CAS(expected, desired, SUCCESS_ORDER, FAILURE_ORDER);
}

// clang-format off
/*

; SM90-PLUS-LABEL: {{[[:space:]]*}}Function : {{.*atomic_compare_exchange.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; MEMBAR-DAG: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-DAG: {{.*}}LD.E.128{{(\.SYS)?}} [[EXPECTED:R[0-9]+]], {{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; BLOCK: {{.*}}ATOM.E.CAS.128.STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, [[OLD:R[0-9]+]], {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, [[EXPECTED]], {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS.128.STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, [[OLD:R[0-9]+]], {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, [[EXPECTED]], {{R[0-9]+}}{{.*}}
; SMXX: {{.*}}ST.E.128{{(\.SYS)?}} {{.*}}, [[OLD]]{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
