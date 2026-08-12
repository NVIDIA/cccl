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
// %PARAM% TEMPLATE,SCOPE,SASS_SCOPE,FILECHECK_PREFIX_SCOPE api cab=ca,tsb,CTA,block:cad=ca,tsd,GPU,non_block:cas=ca,tss,SYS,non_block:carb=car,tsb,CTA,block:card=car,tsd,GPU,non_block:cars=car,tss,SYS,non_block:csa=csa,tss,SYS,non_block:csar=csar,tss,SYS,non_block
// %PARAM% OP,DELTA,FILECHECK_PREFIX_RESULT op post_inc=post_increment,0x1,post_inc:post_dec=post_decrement,0xffffffff,post_dec:pre_inc=pre_increment,0x1,pre_inc:pre_dec=pre_decrement,0xffffffff,pre_dec
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

__device__ auto atomic_increment_decrement(TEMPLATE<int32_t, SCOPE>& atom)
{
  return OP(atom);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_increment_decrement.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SMXX: {{.*}}{{(IMAD\.)?MOV(\.U32)?}} [[OPERAND:R[0-9]+]], {{.*}}[[DELTA]]{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; BLOCK: {{.*}}ATOM.E.ADD.S32.STRONG.{{CTA|SM}} {{P(T|[0-9]+)}}, [[OLD:R[0-9]+]], {{.*}}, [[OPERAND]]{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.ADD.S32.STRONG.[[SASS_SCOPE]] {{P(T|[0-9]+)}}, [[OLD:R[0-9]+]], {{.*}}, [[OPERAND]]{{.*}}
; POST_INC-NOT: {{.*}}{{(VIADD|IADD3)}}{{.*}}[[OLD]]{{.*}}
; POST_DEC-NOT: {{.*}}{{(VIADD|IADD3)}}{{.*}}[[OLD]]{{.*}}
; PRE_INC: {{.*}}{{(VIADD|IADD3)}} R4, [[OLD]], 0x1{{.*}}
; PRE_DEC: {{.*}}{{(VIADD|IADD3)}} R4, [[OLD]], {{-0x1|0xffffffff}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.{{.*}}CAS{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
