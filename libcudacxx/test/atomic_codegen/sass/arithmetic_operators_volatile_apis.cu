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
// %PARAM% TEMPLATE,SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE api vcab=vca,tsb,CTA,block:vcad=vca,tsd,GPU,non_block:vcas=vca,tss,SYS,non_block:vcarb=vcar,tsb,CTA,block:vcard=vcar,tsd,GPU,non_block:vcars=vcar,tss,SYS,non_block:vcsa=vcsa,tss,SYS,non_block:vcsar=vcsar,tss,SYS,non_block
// %PARAM% OP,FILECHECK_PREFIX_RESULT,FILECHECK_PREFIX_OPERAND op post_inc=post_increment,post_inc,inc:post_dec=post_decrement,post_dec,dec:pre_inc=pre_increment,pre_inc,inc:pre_dec=pre_decrement,pre_dec,dec
// clang-format on

#include "atomic_codegen_helpers.h"

template <class Atomic>
__device__ auto post_increment(Atomic& atom)
{
  return atom++;
}

template <class Atomic>
__device__ auto post_decrement(Atomic& atom)
{
  return atom--;
}

template <class Atomic>
__device__ auto pre_increment(Atomic& atom)
{
  return ++atom;
}

template <class Atomic>
__device__ auto pre_decrement(Atomic& atom)
{
  return --atom;
}

extern "C" __device__ auto atomic_codegen_test(TEMPLATE<int32_t, SCOPE>& atom)
{
  return OP(atom);
}

// clang-format off
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.ADD{{(\.S32)?}}{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-DAG: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; NON_BLOCK-DAG: {{.*}}CCTL.IVALL{{.*}}
; INC-DAG: {{.*}}{{(IMAD\.)?MOV(\.U32)?|HFMA2(\.MMA)?}} [[OPERAND:R[0-9]+]], {{.*}}{{(0x1|5\.9604644775390625e-08)([,;[:space:]]|$)}}{{.*}}
; DEC-DAG: {{.*}}{{(IMAD\.)?MOV(\.U32)?}} [[OPERAND:R[0-9]+]], {{.*}}0xffffffff{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.ADD{{(\.S32)?}}{{.*}}
; BLOCK: {{.*}}ATOM.E.ADD{{(\.S32)?}}.STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, [[OLD:R[0-9]+]], {{.*}}, [[OPERAND]]{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.ADD{{(\.S32)?}}.STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, [[OLD:R[0-9]+]], {{.*}}, [[OPERAND]]{{.*}}
; NON_BLOCK-NEXT: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; POST_INC-NOT: {{.*}}{{(VIADD|IADD3|IADD)}}{{.*}}[[OLD]]{{.*}}
; POST_DEC-NOT: {{.*}}{{(VIADD|IADD3|IADD)}}{{.*}}[[OLD]]{{.*}}
; PRE_INC: {{.*}}{{(VIADD|IADD3|IADD)}} {{R[0-9]+}}, {{(PT, PT, )?}}[[OLD]], 0x1{{([,;[:space:]]|$)}}{{.*}}
; PRE_DEC: {{.*}}{{(VIADD|IADD3|IADD)}} {{R[0-9]+}}, {{(PT, PT, )?}}[[OLD]], {{(-0x1|0xffffffff)([,;[:space:]]|$)}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.ADD{{(\.S32)?}}{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
