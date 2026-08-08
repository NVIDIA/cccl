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
// %PARAM% CAS cas compare_exchange_weak:compare_exchange_strong
// %PARAM% SUCCESS_ORDER,FAILURE_ORDER order rr=mor,mor:ar=moa,mor:aa=moa,moa:er=more,mor:br=moar,mor:ba=moar,moa:sr=mosc,mor:sa=mosc,moa:ss=mosc,mosc
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ bool atomic_compare_exchange(cuda::atomic<TYPE, SCOPE>& atom, TYPE& expected, TYPE desired)
{
  return atom.CAS(expected, desired, SUCCESS_ORDER, FAILURE_ORDER);
}

// clang-format off
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_compare_exchange.*}}
; SMXX-NOT: {{.*}}MEMBAR.SC.{{.*}}
; SMXX: {{.*}}BSSY [[SYNC:B[0-9]+]], {{.*}}
; BLOCK: {{.*}}ATOM.E.EXCH.STRONG.{{CTA|SM}} PT, [[LOCK_STATE:R[0-9]+]], {{.*\[}}[[BASE_ADDR:R[0-9]+]]{{(\.64)?\+0x10\].*}}, {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.EXCH.STRONG.[[SASS_SCOPE]] PT, [[LOCK_STATE:R[0-9]+]], {{.*\[}}[[BASE_ADDR:R[0-9]+]]{{(\.64)?\+0x10\].*}}, {{R[0-9]+}}{{.*}}
; SMXX-DAG: {{.*}}ISETP.NE{{(\.U32)?}}.AND [[LOCK_ACQUIRED:P[0-9]+]], PT, [[LOCK_STATE]], 0x1, PT{{.*}}
; NON_BLOCK-DAG: {{.*}}CCTL.IVALL{{.*}}
; SMXX: {{.*}}{{@!?}}[[LOCK_ACQUIRED]] BRA{{(\.U)?}} {{.*}}
; SMXX: {{.*}}BSYNC [[SYNC]]{{.*}}
; SMXX: {{.*}}LD.E.128{{(\.SYS)?}} {{R[0-9]+}}, {{.*\[}}[[BASE_ADDR]]{{(\.64)?\].*}}
; SMXX: {{.*}}LD.E.128{{(\.SYS)?}} {{R[0-9]+}}, {{.*}}
; SMXX: {{.*}}LOP3.LUT [[CAS_RESULT_PRED:P[0-9]+]], RZ, {{R[0-9]+}}, {{R[0-9]+}}, RZ, 0xfc, [[CAS_RESULT_PRED]]{{.*}}
; CUDA12-0: {{.*}}SEL [[STORE_DATA:R[0-9]+]], {{R[0-9]+}}, {{R[0-9]+}}, ![[CAS_RESULT_PRED]]{{.*}}
; CUDA12-0: {{.*}}SEL [[STORE_ADDR:R[0-9]+]], [[BASE_ADDR]], {{R[0-9]+}}, ![[CAS_RESULT_PRED]]{{.*}}
; CUDA12-0: {{.*}}SEL {{R[0-9]+}}, RZ, 0x1, [[CAS_RESULT_PRED]]{{.*}}
; CUDA12-0: {{.*}}ST.E.128{{(\.SYS)?}} {{.*\[}}[[STORE_ADDR]]{{(\.64)?\].*}}, [[STORE_DATA]]{{.*}}
; CUDA12-0-NOT: {{.*}}ST.E.128{{(\.SYS)?}} {{.*}}
; CUDA12-1-PLUS: {{.*}}@[[CAS_RESULT_PRED]] ST.E.128{{(\.SYS)?}} {{.*}}
; CUDA12-1-PLUS: {{.*}}@![[CAS_RESULT_PRED]] ST.E.128{{(\.SYS)?}} {{.*\[}}[[BASE_ADDR]]{{(\.64)?\].*}}
; SMXX: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; BLOCK: {{.*}}ST.E.STRONG.{{CTA|SM}} {{.*\[}}[[BASE_ADDR]]{{(\.64)?\+0x10\].*}}, RZ{{.*}}
; NON_BLOCK: {{.*}}ST.E.STRONG.[[SASS_SCOPE]] {{.*\[}}[[BASE_ADDR]]{{(\.64)?\+0x10\].*}}, RZ{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
