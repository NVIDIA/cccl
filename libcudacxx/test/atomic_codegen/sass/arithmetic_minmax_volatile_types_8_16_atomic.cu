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
// %PARAM% TYPE,SASS_CALC type f16=f16,HSETP2:bf16=bf16,HSETP2
// %PARAM% OP op fetch_min:fetch_max
// %PARAM% ORDER,FILECHECK_PREFIX_ORDER order relaxed=mor,no_membar:acquire=moa,no_membar:release=more,release:acq_rel=moar,release:seq_cst=mosc,seq_cst
// clang-format on

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "atomic_codegen_helpers.h"

__device__ auto atomic_minmax(volatile cuda::atomic<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

// Owning 16-bit extended floating-point atomics use 32-bit storage, so min/max
// must compare the decoded value and update the storage through a 32-bit CAS.
// clang-format off
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_minmax.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; BLOCK-DAG: {{.*}}LD.E.STRONG.{{CTA|SM}} {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR:R[0-9]+]]{{(\.64)?\].*}}
; NON_BLOCK-DAG: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR:R[0-9]+]]{{(\.64)?\].*}}
; SMXX-DAG: {{.*}}[[SASS_CALC]]{{.*}}
; SEQ_CST-DAG: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; RELEASE-DAG: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}LD.E{{.*}}.STRONG{{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS{{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; BLOCK: {{.*}}ATOM.E.CAS.STRONG.{{CTA|SM}} PT, [[OLD:R[0-9]+]], {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, [[EXPECTED:R[0-9]+]], {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS.STRONG.[[SASS_SCOPE]] PT, [[OLD:R[0-9]+]], {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, [[EXPECTED:R[0-9]+]], {{R[0-9]+}}{{.*}}
; SMXX: {{.*}}ISETP.NE{{.*}} [[OLD]], [[EXPECTED]], {{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS{{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; SMXX-NOT: {{.*}}ATOM.E.{{MIN|MAX}}{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
