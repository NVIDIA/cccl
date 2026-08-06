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
// %PARAM% CAS cas compare_exchange_weak:compare_exchange_strong
// %PARAM% SUCCESS_ORDER,FAILURE_ORDER,SASS_MEMBAR,FILECHECK_PREFIX_ORDER order rr=mor,mor,,no_membar:ar=moa,mor,,no_membar:aa=moa,moa,,no_membar:er=more,mor,ALL,membar:br=moar,mor,ALL,membar:ba=moar,moa,ALL,membar:sr=mosc,mor,SC,membar:sa=mosc,moa,SC,membar:ss=mosc,mosc,SC,membar
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ bool atomic_compare_exchange(TEMPLATE<int32_t, SCOPE>& atom, int32_t& expected, int32_t desired)
{
  return atom.CAS(expected, desired, SUCCESS_ORDER, FAILURE_ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_compare_exchange.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; BLOCK: {{.*}}ATOM.E.CAS.STRONG.{{CTA|SM}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS.STRONG.[[SASS_SCOPE]]{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
