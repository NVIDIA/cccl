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
// %PARAM% OP,SASS_CALC op add=fetch_add,IADD3:sub=fetch_sub,IADD3:min=fetch_min,ISETP:max=fetch_max,ISETP
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_ORDER order relaxed=mor,,no_membar:acquire=moa,,no_membar:release=more,ALL,membar:acq_rel=moar,ALL,membar:seq_cst=mosc,SC,membar
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ auto atomic_arithmetic(cuda::atomic_ref<volatile TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

// clang-format off
/*

; SM90-PLUS-LABEL: {{[[:space:]]*}}Function : {{.*atomic_arithmetic.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{ADD|MIN|MAX}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; SMXX-NOT: {{.*}}LD.E.128{{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; BLOCK: {{.*}}LD.E.128.STRONG.{{CTA|SM}} {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; NON_BLOCK: {{.*}}LD.E.128.STRONG.[[SASS_SCOPE]] {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; SMXX-NOT: {{.*}}LD.E.128{{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; SMXX: {{.*}}[[SASS_CALC]]{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{ADD|MIN|MAX}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; BLOCK: {{.*}}ATOM.E.CAS.128.STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, {{R[0-9]+}}, {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS.128.STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, {{R[0-9]+}}, {{R[0-9]+}}{{.*}}
; SMXX: {{.*}}ISETP.NE{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}
; SMXX-NOT: {{.*}}ATOM.E.{{ADD|MIN|MAX}}{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
