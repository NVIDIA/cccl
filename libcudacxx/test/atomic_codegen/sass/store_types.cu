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
// %PARAM% TEMPLATE,SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE api cab=ca,tsb,CTA,block:card=car,tsd,GPU,non_block:csa=csa,tss,SYS,non_block
// %PARAM% TYPE,SASS_SIZE type i32=int32_t,:u32=uint32_t,:i64=int64_t,.64:u64=uint64_t,.64:f32=float,:f64=double,.64:ptr1=char*,.64:ptr4=int32_t*,.64
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_ORDER order relaxed=mor,,no_membar:release=more,ALL,membar:seq_cst=mosc,SC,membar
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ void atomic_store(TEMPLATE<TYPE, SCOPE>& atom, TYPE value)
{
  atom.store(value, ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_store.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; BLOCK: {{.*}}ST.E[[SASS_SIZE]].STRONG.{{CTA|SM}} {{.*}}, R6{{.*}}
; NON_BLOCK: {{.*}}ST.E[[SASS_SIZE]].STRONG.[[SASS_SCOPE]] {{.*}}, R6{{.*}}
; SMXX-NEXT: {{.*}}RET.ABS.NODEC{{.*}}

*/
