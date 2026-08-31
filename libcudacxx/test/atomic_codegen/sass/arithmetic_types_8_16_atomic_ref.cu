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
// %PARAM% OP,FILECHECK_PREFIX_OP op add=fetch_add,add:sub=fetch_sub,sub:min=fetch_min,min:max=fetch_max,max
// %PARAM% ORDER,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,non_seq_cst,no_acquire,no_membar:acquire=moa,non_seq_cst,acquire,no_membar:release=more,non_seq_cst,no_acquire,release:acq_rel=moar,non_seq_cst,acquire,release:seq_cst=mosc,seq_cst,acquire,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// %FILECHECK% PREFIX_COMBINE ptx,seq_cst
// %FILECHECK% PREFIX_COMBINE ptx,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE ptx,release
// %FILECHECK% PREFIX_COMBINE sm75,nvvm
// %FILECHECK% PREFIX_COMBINE sm75,nvvm,seq_cst
// %FILECHECK% PREFIX_COMBINE sm75,nvvm,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE sm75,nvvm,release
// %FILECHECK% PREFIX_COMBINE sm80,nvvm
// %FILECHECK% PREFIX_COMBINE sm80,nvvm,seq_cst
// %FILECHECK% PREFIX_COMBINE sm80,nvvm,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE sm80,nvvm,release
// %FILECHECK% PREFIX_COMBINE sm90,nvvm
// %FILECHECK% PREFIX_COMBINE sm90,nvvm,seq_cst
// %FILECHECK% PREFIX_COMBINE sm90,nvvm,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE sm90,nvvm,release
// %FILECHECK% PREFIX_COMBINE sm100,nvvm,seq_cst
// %FILECHECK% PREFIX_COMBINE sm100,nvvm,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE sm100,nvvm,release
// %FILECHECK% PREFIX_COMBINE sm120,nvvm,seq_cst
// %FILECHECK% PREFIX_COMBINE sm120,nvvm,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE sm120,nvvm,release
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(cuda::atomic_ref<TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

// clang-format off
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.E.{{ADD|MIN|MAX}}{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; SM100_NVVM_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SM120_NVVM_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SM100_NVVM_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SM120_NVVM_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SM100_NVVM_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; SM120_NVVM_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-DAG: {{.*}}LOP3.LUT [[ALIGNED_ADDR:R[0-9]+]], [[ATOM_ADDR]]{{(\.reuse)?}}, 0xfffffffc, {{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; BLOCK: {{.*}}LD.E.STRONG.{{CTA|SM}} [[EXPECTED:R[0-9]+]], {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; NON_BLOCK: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] [[EXPECTED:R[0-9]+]], {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; ADD-NOT: {{.*}}{{IADD3|IADD|IMAD\.IADD|IMAD\.MOV}} {{R[0-9]+}}, {{.*}}-{{R[0-9]+}}{{.*}}
; ADD: {{.*}}{{IADD3|IADD|IMAD\.IADD}}{{.*}}
; SUB: {{.*}}{{IADD3|IADD|IMAD\.IADD|IMAD\.MOV}} [[SUB_VALUE:R[0-9]+]], {{.*}}-{{R[0-9]+}}{{.*}}
; SUB: {{.*}}{{IADD3|IADD|IMAD\.IADD|PRMT|LOP3\.LUT}} {{R[0-9]+}}, {{.*}}[[SUB_VALUE]]{{.*}}
; MIN: {{.*}}IMNMX{{.*}}
; MAX: {{.*}}IMNMX{{.*}}
; PTX_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SM75_NVVM_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SM80_NVVM_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SM90_NVVM_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; PTX_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SM75_NVVM_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SM80_NVVM_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SM90_NVVM_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; PTX_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; SM75_NVVM_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; SM80_NVVM_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; SM90_NVVM_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{ADD|MIN|MAX}}{{.*}}
; BLOCK: {{.*}}ATOM.E.CAS.STRONG.{{CTA|SM}} PT, [[OLD:R[0-9]+]], {{\[}}[[ALIGNED_ADDR]]{{\]}}, [[EXPECTED]], {{R[0-9]+}}{{.*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS.STRONG.[[SASS_SCOPE]] PT, [[OLD:R[0-9]+]], {{\[}}[[ALIGNED_ADDR]]{{\]}}, [[EXPECTED]], {{R[0-9]+}}{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; PTX-DAG: {{.*}}ISETP.NE{{.*}} [[OLD]]{{(\.reuse)?}}, [[EXPECTED]]{{(\.reuse)?}}, {{.*}}
; SM75_NVVM-DAG: {{.*}}ISETP.NE{{.*}} [[EXPECTED]]{{(\.reuse)?}}, [[OLD]]{{(\.reuse)?}}, {{.*}}
; SM80_NVVM-DAG: {{.*}}ISETP.NE{{.*}} [[EXPECTED]]{{(\.reuse)?}}, [[OLD]]{{(\.reuse)?}}, {{.*}}
; SM90_NVVM-DAG: {{.*}}ISETP.NE{{.*}} [[EXPECTED]]{{(\.reuse)?}}, [[OLD]]{{(\.reuse)?}}, {{.*}}
; NON_BLOCK_ACQUIRE-DAG: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.{{ADD|MIN|MAX}}{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.CAS{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
