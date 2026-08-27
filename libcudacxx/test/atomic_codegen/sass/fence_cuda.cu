//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// clang-format off
// %PARAM% SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE scope block=tsb,CTA,block:device=tsd,GPU,non_block:system=tss,SYS,non_block
// %PARAM% ORDER,SASS_SEMANTIC order acquire=moa,ALL:release=more,ALL:acq_rel=moar,ALL:seq_cst=mosc,SC
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ int32_t atomic_codegen_test(int32_t* before, const int32_t* after, int32_t value)
{
  *before = value;
  cuda::atomic_thread_fence(ORDER, SCOPE);
  return *after;
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX: {{.*}}ST{{G?}}.E{{.*}}
; SMXX-NOT: {{.*}}ST{{G?}}.E{{.*}}
; SMXX-NOT: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}MEMBAR.{{.*}}
; NON_BLOCK-NOT: {{.*}}MEMBAR.{{.*}}.[[SASS_SCOPE]]{{.*}}
; SMXX: {{.*}}MEMBAR.[[SASS_SEMANTIC]].[[SASS_SCOPE]]{{.*}}
; BLOCK-NOT: {{.*}}MEMBAR.{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_BLOCK-NOT: {{.*}}MEMBAR.{{.*}}
; NON_BLOCK: {{.*}}CCTL.IVALL{{.*}}
; NON_BLOCK-NOT: {{.*}}MEMBAR.{{.*}}
; NON_BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX: {{.*}}LD{{G?}}.E{{.*}}
; SMXX-NOT: {{.*}}LD{{G?}}.E{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
