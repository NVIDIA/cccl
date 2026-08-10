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
// %PARAM% SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE scope device=tsd,GPU,non_block
// %PARAM% TYPE type int8_t:uint8_t:int16_t:uint16_t:f16:bf16
// Strong compare-exchange may retry internally using weak compare-exchange; only the weak overload must contain one CAS.
// %PARAM% CAS,FILECHECK_PREFIX_SINGLE_CAS cas compare_exchange_weak=compare_exchange_weak,single_cas:compare_exchange_strong=compare_exchange_strong,smxx
// %PARAM% SUCCESS_ORDER,FAILURE_ORDER,SASS_MEMBAR,FILECHECK_PREFIX_ORDER order rr=mor,mor,,no_membar:ar=moa,mor,,no_membar:aa=moa,moa,,no_membar:er=more,mor,ALL,membar:br=moar,mor,ALL,membar:ba=moar,moa,ALL,membar:sr=mosc,mor,SC,membar:sa=mosc,moa,SC,membar:ss=mosc,mosc,SC,membar
// clang-format on

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "atomic_codegen_helpers.h"

__device__ bool atomic_compare_exchange(volatile cuda::atomic<TYPE, SCOPE>& atom, TYPE& expected, TYPE desired)
{
  return atom.CAS(expected, desired, SUCCESS_ORDER, FAILURE_ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_compare_exchange.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS{{.*}}
; BLOCK: {{.*}}ATOM.E.CAS.STRONG.{{CTA|SM}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS.STRONG.[[SASS_SCOPE]]{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SINGLE_CAS-NOT: {{.*}}ATOM.E.CAS{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
