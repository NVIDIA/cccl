//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// %PARAM% ORDER,SASS_SEMANTIC order acquire=moa,ALL:release=more,ALL:acq_rel=moar,ALL:seq_cst=mosc,SC

#include "atomic_codegen_helpers.h"

extern "C" __device__ int32_t atomic_codegen_test(int32_t* before, const int32_t* after, int32_t value)
{
  *before = value;
  cuda::std::atomic_thread_fence(ORDER);
  return *after;
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX: {{.*}}ST{{G?}}.E{{.*}}
; SMXX-NOT: {{.*}}ST{{G?}}.E{{.*}}
; SMXX-NOT: {{.*}}MEMBAR.{{.*}}.SYS{{.*}}
; SMXX-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX: {{.*}}MEMBAR.[[SASS_SEMANTIC]].SYS{{.*}}
; SMXX-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX: {{.*}}LD{{G?}}.E{{.*}}
; SMXX-NOT: {{.*}}LD{{G?}}.E{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
