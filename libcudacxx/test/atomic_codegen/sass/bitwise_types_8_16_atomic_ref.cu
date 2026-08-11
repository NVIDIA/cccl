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
// %PARAM% TYPE type int8_t:uint8_t:int16_t:uint16_t
// %PARAM% OP,SASS_OP op and=fetch_and,AND:or=fetch_or,OR:xor=fetch_xor,XOR
// %PARAM% ORDER,FILECHECK_PREFIX_ORDER order relaxed=mor,no_membar:acquire=moa,no_membar:release=more,release:acq_rel=moar,release:seq_cst=mosc,seq_cst
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ auto atomic_bitwise(cuda::atomic_ref<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

// The compiler may fuse the bitwise operation with subword packing, so check
// the CAS protocol without fixing a particular LOP3 truth table.
// clang-format off
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_bitwise.*}}
; SMXX-NOT: {{.*}}ATOM.E.[[SASS_OP]]{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; SMXX-DAG: {{.*}}LOP3.LUT [[ALIGNED_ADDR:R[0-9]+]], [[ATOM_ADDR]]{{(\.reuse)?}}, 0xfffffffc, {{.*}}
; SEQ_CST-DAG: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; BLOCK: {{.*}}LD.E.STRONG.{{CTA|SM}} [[EXPECTED:R[0-9]+]], {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; NON_BLOCK: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] [[EXPECTED:R[0-9]+]], {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; SMXX: {{.*}}LOP3.LUT {{.*}}
; RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.[[SASS_OP]]{{.*}}
; BLOCK: {{.*}}ATOM.E.CAS.STRONG.{{CTA|SM}} PT, [[OLD:R[0-9]+]], {{\[}}[[ALIGNED_ADDR]]{{\]}}, [[EXPECTED]], {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS.STRONG.[[SASS_SCOPE]] PT, [[OLD:R[0-9]+]], {{\[}}[[ALIGNED_ADDR]]{{\]}}, [[EXPECTED]], {{R[0-9]+}}{{.*}}
; SMXX: {{.*}}ISETP.NE{{.*}} [[OLD]], [[EXPECTED]], {{.*}}
; SMXX-NOT: {{.*}}ATOM.E.[[SASS_OP]]{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
