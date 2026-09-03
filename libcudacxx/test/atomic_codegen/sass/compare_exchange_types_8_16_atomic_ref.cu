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
// %PARAM% CAS cas compare_exchange_weak:compare_exchange_strong
// %PARAM% SUCCESS_ORDER,FAILURE_ORDER,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order rr=mor,mor,non_seq_cst,no_acquire,no_membar:ar=moa,mor,non_seq_cst,acquire,no_membar:aa=moa,moa,non_seq_cst,acquire,no_membar:er=more,mor,non_seq_cst,no_acquire,release:br=moar,mor,non_seq_cst,acquire,release:ba=moar,moa,non_seq_cst,acquire,release:sr=mosc,mor,seq_cst,acquire,seq_cst:sa=mosc,moa,seq_cst,acquire,seq_cst:ss=mosc,mosc,seq_cst,acquire,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// clang-format on

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "atomic_codegen_helpers.h"

extern "C" __device__ bool atomic_codegen_test(cuda::atomic_ref<TYPE, SCOPE>& atom, TYPE& expected, TYPE desired)
{
  return atom.CAS(expected, desired, SUCCESS_ORDER, FAILURE_ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-DAG: {{.*}}LOP3.LUT [[ALIGNED_ADDR:R[0-9]+]], [[ATOM_ADDR]]{{(\.reuse)?}}, 0xfffffffc, {{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; BLOCK: {{.*}}LD.E.STRONG.{{CTA|SM}} {{R[0-9]+}}, {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; NON_BLOCK: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] {{R[0-9]+}}, {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; BLOCK: {{.*}}ATOM.E.CAS.STRONG.{{CTA|SM}}{{.*\[}}[[ALIGNED_ADDR]]{{\].*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS.STRONG.[[SASS_SCOPE]]{{.*\[}}[[ALIGNED_ADDR]]{{\].*}}
; NON_BLOCK_ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
