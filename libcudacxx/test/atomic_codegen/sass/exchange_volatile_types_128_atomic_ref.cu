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
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_ORDER order relaxed=mor,,no_membar:acquire=moa,,no_membar:release=more,ALL,membar:acq_rel=moar,ALL,membar:seq_cst=mosc,SC,membar
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ auto atomic_exchange(cuda::atomic_ref<volatile TYPE, SCOPE>& atom, TYPE value)
{
  return atom.exchange(value, ORDER);
}

/*

; SM90-PLUS-LABEL: {{[[:space:]]*}}Function : {{.*atomic_exchange.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH.STRONG{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; SMXX-NOT: {{.*}}ATOM.E{{.*\[}}[[ATOM_ADDR]]{{(\.64)?(\+0x[0-9a-f]+)?\].*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E{{.*\[}}[[ATOM_ADDR]]{{(\.64)?(\+0x[0-9a-f]+)?\].*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH.STRONG{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; BLOCK: {{.*}}ATOM.E.EXCH.128.STRONG.{{CTA|SM}} PT, R4, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, R8{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.EXCH.128.STRONG.[[SASS_SCOPE]] PT, R4, {{.*\[}}[[ATOM_ADDR]]{{(\.64)?\].*}}, R8{{.*}}
; SMXX-NOT: {{.*}}ATOM.E{{.*\[}}[[ATOM_ADDR]]{{(\.64)?(\+0x[0-9a-f]+)?\].*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
