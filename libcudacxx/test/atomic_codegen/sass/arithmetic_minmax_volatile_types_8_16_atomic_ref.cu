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
// %PARAM% SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE scope device=tsd,GPU,non_block
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

extern "C" __device__ auto atomic_codegen_test(cuda::atomic_ref<volatile TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

// clang-format off
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-DAG: {{.*}}LOP3.LUT [[ALIGNED_ADDR:R[0-9]+]], [[ATOM_ADDR]]{{(\.reuse)?}}, 0xfffffffc, {{.*}}
; F16-DAG: {{.*}}HSETP2{{.*}}
; NOT-SM90-PLUS_BF16-DAG: {{.*}}FSETP{{.*}}
; SM90-PLUS_BF16-DAG: {{.*}}HSETP2{{.*}}
; BLOCK-DAG: {{.*}}LD.E.STRONG.{{CTA|SM}} [[EXPECTED:R[0-9]+]], {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; NON_BLOCK-DAG: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] [[EXPECTED:R[0-9]+]], {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; RELEASE-DAG: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS{{.*\[}}[[ALIGNED_ADDR]]{{\].*}}
; BLOCK: {{.*}}ATOM.E.CAS.STRONG.{{CTA|SM}} PT, [[OLD:R[0-9]+]], {{\[}}[[ALIGNED_ADDR]]{{\]}}, [[EXPECTED]], {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS.STRONG.[[SASS_SCOPE]] PT, [[OLD:R[0-9]+]], {{\[}}[[ALIGNED_ADDR]]{{\]}}, [[EXPECTED]], {{R[0-9]+}}{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; CAS_COMPARE_OLD_EXPECTED-DAG: {{.*}}ISETP.NE{{.*}} [[OLD]]{{(\.reuse)?}}, [[EXPECTED]]{{(\.reuse)?}}, {{.*}}
; CAS_COMPARE_EXPECTED_OLD-DAG: {{.*}}ISETP.NE{{.*}} [[EXPECTED]]{{(\.reuse)?}}, [[OLD]]{{(\.reuse)?}}, {{.*}}
; NON_BLOCK_ACQUIRE-DAG: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS{{.*\[}}[[ALIGNED_ADDR]]{{\].*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
