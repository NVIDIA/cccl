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
// %PARAM% TYPE,SASS_SIZE type i32=int32_t,:u32=uint32_t,:i64=int64_t,.64:u64=uint64_t,.64:f32=float,:f64=double,.64:ptr1=char*,.64:ptr4=int32_t*,.64
// %PARAM% CAS cas compare_exchange_weak:compare_exchange_strong
// %PARAM% SUCCESS_ORDER,FAILURE_ORDER,SASS_MEMBAR,FILECHECK_PREFIX_ORDER order rr=mor,mor,,no_membar:ar=moa,mor,,no_membar:aa=moa,moa,,no_membar:er=more,mor,ALL,membar:br=moar,mor,ALL,membar:ba=moar,moa,ALL,membar:sr=mosc,mor,SC,membar:sa=mosc,moa,SC,membar:ss=mosc,mosc,SC,membar
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ bool atomic_compare_exchange(TEMPLATE<TYPE, SCOPE>& atom, TYPE& expected, TYPE desired)
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
; BLOCK: {{.*}}ATOM.E.CAS[[SASS_SIZE]].STRONG.{{CTA|SM}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS[[SASS_SIZE]].STRONG.[[SASS_SCOPE]]{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
