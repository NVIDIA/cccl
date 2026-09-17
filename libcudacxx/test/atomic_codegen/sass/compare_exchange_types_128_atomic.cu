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
// %FILECHECK% PREFIX_COMBINE sm75,cuda12-0
// %FILECHECK% PREFIX_COMBINE sm80,cuda12-0,block
// %FILECHECK% PREFIX_COMBINE sm80,cuda12-0,non_block
// %FILECHECK% PREFIX_COMBINE sm90,cuda12-0
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ bool atomic_codegen_test(cuda::atomic<TYPE, SCOPE>& atom, TYPE& expected, TYPE desired)
{
  return atom.CAS(expected, desired, SUCCESS_ORDER, FAILURE_ORDER);
}

// clang-format off
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}MEMBAR.SC.{{.*}}
; SMXX: {{.*}}BSSY [[SYNC:B[0-9]+]], {{.*}}
; BLOCK: {{.*}}ATOM.E.EXCH.STRONG.{{CTA|SM}} PT, [[LOCK_STATE:R[0-9]+]], {{.*\[}}[[BASE_ADDR:R[0-9]+]]{{(\.64)?\+0x10\].*}}, {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.EXCH.STRONG.[[SASS_SCOPE]] PT, [[LOCK_STATE:R[0-9]+]], {{.*\[}}[[BASE_ADDR:R[0-9]+]]{{(\.64)?\+0x10\].*}}, {{R[0-9]+}}{{.*}}
; SMXX-DAG: {{.*}}ISETP.NE{{(\.U32)?}}.AND [[LOCK_ACQUIRED:P[0-9]+]], PT, [[LOCK_STATE]], 0x1, PT{{.*}}
; NON_BLOCK-DAG: {{.*}}CCTL.IVALL{{.*}}
; SM75: {{.*}}@[[LOCK_ACQUIRED]] BRA{{(\.U)?}} {{.*}}
; SM75: {{.*}}LOP3.LUT [[RETRY_PRED:P[0-9]+]], {{.*}}
; SM75: {{.*}}@![[RETRY_PRED]] BRA{{(\.U)?}} {{.*}}
; SM80-PLUS: {{.*}}@![[LOCK_ACQUIRED]] BRA{{(\.U)?}} {{.*}}
; SMXX: {{.*}}BSYNC [[SYNC]]{{.*}}
; SMXX: {{.*}}LD.E.128{{(\.SYS)?}} {{R[0-9]+}}, {{.*\[}}[[BASE_ADDR]]{{(\.64)?\].*}}
; SMXX: {{.*}}LD.E.128{{(\.SYS)?}} {{R[0-9]+}}, {{.*}}
; SMXX: {{.*}}{{LOP3\.LUT|ISETP\.NE[^ ]*}} [[CAS_RESULT_PRED:P[0-9]+]], {{.*}}
; CUDA12-0-DAG: {{.*}}SEL [[STORE_DATA:R[0-9]+]], {{R[0-9]+}}, {{R[0-9]+}}, ![[CAS_RESULT_PRED]]{{.*}}
; CUDA12-0-DAG: {{.*}}SEL [[STORE_ADDR:R[0-9]+]], [[BASE_ADDR]], {{R[0-9]+}}, ![[CAS_RESULT_PRED]]{{.*}}
; SM90_CUDA12-0-DAG: {{.*}}SEL {{R[0-9]+}}, RZ, 0x1, [[CAS_RESULT_PRED]]{{.*}}
; CUDA12-0-DAG: {{.*}}ST.E.128{{(\.SYS)?}} {{.*\[}}[[STORE_ADDR]]{{(\.64)?\].*}}, [[STORE_DATA]]{{.*}}
; CUDA12-1-PLUS-DAG: {{.*}}@[[CAS_RESULT_PRED]] ST.E.128{{(\.SYS)?}} {{.*}}
; CUDA12-1-PLUS-DAG: {{.*}}@![[CAS_RESULT_PRED]] ST.E.128{{(\.SYS)?}} {{.*\[}}[[BASE_ADDR]]{{(\.64)?\].*}}
; SMXX: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SM80_CUDA12-0_BLOCK: {{.*}}SEL {{R[0-9]+}}, RZ, 0x1, [[CAS_RESULT_PRED]]{{.*}}
; BLOCK: {{.*}}ST.E.STRONG.{{CTA|SM}} {{.*\[}}[[BASE_ADDR]]{{(\.64)?\+0x10\].*}}, RZ{{.*}}
; NON_BLOCK: {{.*}}ST.E.STRONG.[[SASS_SCOPE]] {{.*\[}}[[BASE_ADDR]]{{(\.64)?\+0x10\].*}}, RZ{{.*}}
; SM75_CUDA12-0: {{.*}}SEL {{R[0-9]+}}, RZ, 0x1, [[CAS_RESULT_PRED]]{{.*}}
; SM80_CUDA12-0_NON_BLOCK: {{.*}}SEL {{R[0-9]+}}, RZ, 0x1, [[CAS_RESULT_PRED]]{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
