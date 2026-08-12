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
// %PARAM% TYPE,SASS_TYPE type i8=int8_t,.S32:u8=uint8_t,:i16=int16_t,.S32:u16=uint16_t,
// %PARAM% OP,SASS_OP op add=fetch_add,ADD:sub=fetch_sub,ADD:min=fetch_min,MIN:max=fetch_max,MAX
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_ORDER order relaxed=mor,,no_membar:acquire=moa,,no_membar:release=more,ALL,membar:acq_rel=moar,ALL,membar:seq_cst=mosc,SC,membar
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ auto atomic_arithmetic(volatile cuda::atomic<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_arithmetic.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.[[SASS_OP]][[SASS_TYPE]]{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.[[SASS_OP]][[SASS_TYPE]]{{.*}}
; BLOCK: {{.*}}ATOM.E.[[SASS_OP]][[SASS_TYPE]].STRONG.{{CTA|SM}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.[[SASS_OP]][[SASS_TYPE]].STRONG.[[SASS_SCOPE]]{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.[[SASS_OP]][[SASS_TYPE]]{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
