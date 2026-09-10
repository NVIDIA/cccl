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
// %PARAM% ORDER,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,no_acquire,non_sc:acquire=moa,acquire,non_sc:seq_cst=mosc,acquire,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(TEMPLATE<int32_t, SCOPE>& atom)
{
  return atom.load(ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SC-NOT: {{.*}}CCTL.IVALL{{.*}}
; SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SC-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SC-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; BLOCK: {{.*}}LD.E.STRONG.{{CTA|SM}} R4, {{.*}}
; NON_BLOCK: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] R4, {{.*}}
; NON_BLOCK_ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
