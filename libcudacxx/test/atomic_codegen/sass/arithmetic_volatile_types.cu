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
// %PARAM% TYPE,SASS_TYPE type i32=int32_t,.S32:u32=uint32_t,:i64=int64_t,.64:u64=uint64_t,.64
// %PARAM% OP op fetch_add:fetch_sub
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_ORDER order relaxed=mor,,no_membar:acquire=moa,,no_membar:release=more,ALL,membar:acq_rel=moar,ALL,membar:seq_cst=mosc,SC,membar
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ auto atomic_arithmetic(TEMPLATE<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_arithmetic.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.ADD[[SASS_TYPE]]{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.ADD[[SASS_TYPE]]{{.*}}
; BLOCK: {{.*}}ATOM.E.ADD[[SASS_TYPE]].STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.ADD[[SASS_TYPE]].STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.ADD[[SASS_TYPE]]{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
