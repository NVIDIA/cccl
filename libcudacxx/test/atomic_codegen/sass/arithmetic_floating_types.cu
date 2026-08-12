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
// %PARAM% TEMPLATE,SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE api cab=ca,tsb,CTA,block:card=car,tsd,GPU,non_block:csa=csa,tss,SYS,non_block
// %PARAM% TYPE,SASS_TYPE type f32=float,.F32.FTZ.RN:f64=double,.F64.RN
// %PARAM% OP op fetch_add:fetch_sub
// %PARAM% ORDER,SASS_MEMBAR,FILECHECK_PREFIX_ORDER order relaxed=mor,,no_membar:acquire=moa,,no_membar:release=more,ALL,membar:acq_rel=moar,ALL,membar:seq_cst=mosc,SC,membar
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ auto atomic_arithmetic(TEMPLATE<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

// Generic-address floating-point atomics contain a separate CAS fallback for
// local memory. The native path must still use the floating-point add opcode.
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_arithmetic.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].[[SASS_SCOPE]]{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; BLOCK: {{.*}}ATOM.E.ADD[[SASS_TYPE]].STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.ADD[[SASS_TYPE]].STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, {{R[0-9]+}}, {{.*}}, {{R[0-9]+}}{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
