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
// %PARAM% TYPE,FILECHECK_PREFIX_TYPE type f16=f16,f16:bf16=bf16,bf16
// %PARAM% OP op fetch_min:fetch_max
// %PARAM% ORDER,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,non_seq_cst,no_acquire,no_membar:acquire=moa,non_seq_cst,acquire,no_membar:release=more,non_seq_cst,no_acquire,release:acq_rel=moar,non_seq_cst,acquire,release:seq_cst=mosc,seq_cst,acquire,seq_cst
// %FILECHECK% PREFIX_COMBINE not-sm90-plus,bf16
// %FILECHECK% PREFIX_COMBINE sm90-plus,bf16
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// clang-format on

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(cuda::atomic<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

// Owning 16-bit extended floating-point atomics use 32-bit storage, so min/max
// must compare the decoded value and update the storage through a 32-bit CAS.
// clang-format off
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-DAG: {{.*}}LD.E.STRONG.{{CTA|SM}} {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR:R[0-9]+]]{{(\.64)?\].*}}
; NON_BLOCK-DAG: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR:R[0-9]+]]{{(\.64)?\].*}}
; F16-DAG: {{.*}}HSETP2{{.*}}
; NOT-SM90-PLUS_BF16-DAG: {{.*}}FSETP{{.*}}
; SM90-PLUS_BF16-DAG: {{.*}}HSETP2{{.*}}
; SEQ_CST-DAG: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST-DAG: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; RELEASE-DAG: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*}}.STRONG{{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS{{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; BLOCK: {{.*}}ATOM.E.CAS.STRONG.{{CTA|SM}} PT, [[OLD:R[0-9]+]], {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, [[EXPECTED:R[0-9]+]], {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS.STRONG.[[SASS_SCOPE]] PT, [[OLD:R[0-9]+]], {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, [[EXPECTED:R[0-9]+]], {{R[0-9]+}}{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; CAS_COMPARE_OLD_EXPECTED-DAG: {{.*}}ISETP.NE{{.*}} [[OLD]]{{(\.reuse)?}}, [[EXPECTED]]{{(\.reuse)?}}, {{.*}}
; CAS_COMPARE_EXPECTED_OLD-DAG: {{.*}}ISETP.NE{{.*}} [[EXPECTED]]{{(\.reuse)?}}, [[OLD]]{{(\.reuse)?}}, {{.*}}
; NON_BLOCK_ACQUIRE-DAG: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS{{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
