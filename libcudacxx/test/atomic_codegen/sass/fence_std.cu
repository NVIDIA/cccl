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
// %PARAM% ORDER,SASS_SEMANTIC,FILECHECK_PREFIX_MEMBAR,FILECHECK_PREFIX_ACQUIRE order acquire=moa,ALL,no_membar,acquire:release=more,ALL,membar,no_acquire:acq_rel=moar,ALL,membar,acquire:seq_cst=mosc,SC,membar,acquire
// %FILECHECK% PREFIX_COMBINE ptx,no_membar
// %FILECHECK% PREFIX_COMBINE ptx,no_acquire
// %FILECHECK% PREFIX_COMBINE not-sm90-plus,nvvm,no_membar
// %FILECHECK% PREFIX_COMBINE not-sm90-plus,nvvm,no_acquire
// %FILECHECK% PREFIX_COMBINE sm90-plus,nvvm,no_membar
// %FILECHECK% PREFIX_COMBINE sm90-plus,nvvm,no_acquire
// clang-format on

#include <cuda/std/cstdint>

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
; MEMBAR: {{.*}}MEMBAR.[[SASS_SEMANTIC]].SYS{{.*}}
; PTX_NO_MEMBAR: {{.*}}MEMBAR.[[SASS_SEMANTIC]].SYS{{.*}}
; NOT-SM90-PLUS_NVVM_NO_MEMBAR: {{.*}}MEMBAR.[[SASS_SEMANTIC]].SYS{{.*}}
; SM90-PLUS_NVVM_NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; PTX_NO_ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; NOT-SM90-PLUS_NVVM_NO_ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; SM90-PLUS_NVVM_NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX: {{.*}}LD{{G?}}.E{{.*}}
; SMXX-NOT: {{.*}}LD{{G?}}.E{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
