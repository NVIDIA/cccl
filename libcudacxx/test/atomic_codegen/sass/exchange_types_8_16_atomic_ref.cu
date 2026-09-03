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
// %PARAM% TYPE type int8_t:uint8_t:int16_t:uint16_t:f16:bf16
// %PARAM% ORDER,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,non_seq_cst,no_acquire,no_membar:acquire=moa,non_seq_cst,acquire,no_membar:release=more,non_seq_cst,no_acquire,release:acq_rel=moar,non_seq_cst,acquire,release:seq_cst=mosc,seq_cst,acquire,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// %FILECHECK% PREFIX_COMBINE subword_rmw_fence_late,seq_cst
// %FILECHECK% PREFIX_COMBINE subword_rmw_fence_late,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE subword_rmw_fence_late,release
// %FILECHECK% PREFIX_COMBINE subword_rmw_fence_early,seq_cst
// %FILECHECK% PREFIX_COMBINE subword_rmw_fence_early,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE subword_rmw_fence_early,release
// clang-format on

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(cuda::atomic_ref<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.exchange(value, ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; SUBWORD_RMW_FENCE_EARLY_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SUBWORD_RMW_FENCE_EARLY_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SUBWORD_RMW_FENCE_EARLY_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-DAG: {{.*}}LOP3.LUT [[A:R[0-9]+]], [[ATOM_ADDR]], 0xfffffffc, {{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; BLOCK: {{.*}}LD.E.STRONG.{{CTA|SM}} [[E:R[0-9]+]], {{.*\[}}[[A]]{{(\.64)?\].*}}
; NON_BLOCK: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] [[E:R[0-9]+]], {{.*\[}}[[A]]{{(\.64)?\].*}}
; SUBWORD_RMW_FENCE_LATE_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SUBWORD_RMW_FENCE_LATE_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SUBWORD_RMW_FENCE_LATE_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; BLOCK: {{.*}}ATOM.E.CAS.STRONG.{{CTA|SM}} PT, [[C:R[0-9]+]], {{\[}}[[A]]{{\]}}, [[E]], {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS.STRONG.[[SASS_SCOPE]] PT, [[C:R[0-9]+]], {{\[}}[[A]]{{\]}}, [[E]], {{R[0-9]+}}{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_BLOCK_ACQUIRE-DAG: {{.*}}CCTL.IVALL{{.*}}
; CAS_COMPARE_OLD_EXPECTED-DAG: {{.*}}ISETP.NE{{.*}} [[C]]{{(\.reuse)?}}, [[E]]{{(\.reuse)?}}, {{.*}}
; CAS_COMPARE_EXPECTED_OLD-DAG: {{.*}}ISETP.NE{{.*}} [[E]]{{(\.reuse)?}}, [[C]]{{(\.reuse)?}}, {{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
