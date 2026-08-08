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
// %PARAM% SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE scope block=tsb,CTA,block:device=tsd,GPU,non_block:system=tss,SYS,non_block
// %PARAM% TYPE type i128:u128
// %PARAM% ORDER order mor:moa:more:moar:mosc
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ auto atomic_exchange(cuda::atomic<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.exchange(value, ORDER);
}

// clang-format off
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_exchange.*}}
; SMXX-NOT: {{.*}}MEMBAR.SC.{{.*}}
; SMXX: {{.*}}BSSY [[SYNC:B[0-9]+]], {{.*}}
; BLOCK: {{.*}}ATOM.E.EXCH.STRONG.{{CTA|SM}} PT, [[LOCK_STATE:R[0-9]+]], {{.*\[}}[[BASE_ADDR:R[0-9]+]]{{(\.64)?\+0x10\].*}}, {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.EXCH.STRONG.[[SASS_SCOPE]] PT, [[LOCK_STATE:R[0-9]+]], {{.*\[}}[[BASE_ADDR:R[0-9]+]]{{(\.64)?\+0x10\].*}}, {{R[0-9]+}}{{.*}}
; SMXX-DAG: {{.*}}ISETP.NE{{(\.U32)?}}.AND [[LOCK_FREE:P[0-9]+]], PT, [[LOCK_STATE]], 0x1, PT{{.*}}
; NON_BLOCK-DAG: {{.*}}CCTL.IVALL{{.*}}
; SMXX: {{.*}}{{@!?}}[[LOCK_FREE]] BRA{{(\.U)?}} {{.*}}
; SMXX: {{.*}}BSYNC [[SYNC]]{{.*}}
; SMXX: {{.*}}LD.E.128{{(\.SYS)?}} R4, {{.*\[}}[[BASE_ADDR]]{{(\.64)?\].*}}
; SMXX: {{.*}}ST.E.128{{(\.SYS)?}} {{.*\[}}[[BASE_ADDR]]{{(\.64)?\].*}}, R8{{.*}}
; SMXX: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; BLOCK: {{.*}}ST.E.STRONG.{{CTA|SM}} {{.*\[}}[[BASE_ADDR]]{{(\.64)?\+0x10\].*}}, RZ{{.*}}
; NON_BLOCK: {{.*}}ST.E.STRONG.[[SASS_SCOPE]] {{.*\[}}[[BASE_ADDR]]{{(\.64)?\+0x10\].*}}, RZ{{.*}}
; SMXX-NEXT: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
