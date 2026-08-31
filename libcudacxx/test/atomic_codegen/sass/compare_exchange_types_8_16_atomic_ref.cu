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
// %PARAM% TYPE,FILECHECK_PREFIX_WIDTH type int8_t=int8_t,byte:uint8_t=uint8_t,byte:int16_t=int16_t,halfword:uint16_t=uint16_t,halfword:f16=f16,halfword:bf16=bf16,halfword
// %PARAM% CAS cas compare_exchange_weak:compare_exchange_strong
// %PARAM% SUCCESS_ORDER,FAILURE_ORDER,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order rr=mor,mor,non_seq_cst,no_acquire,no_membar:ar=moa,mor,non_seq_cst,acquire,no_membar:aa=moa,moa,non_seq_cst,acquire,no_membar:er=more,mor,non_seq_cst,no_acquire,release:br=moar,mor,non_seq_cst,acquire,release:ba=moar,moa,non_seq_cst,acquire,release:sr=mosc,mor,seq_cst,acquire,seq_cst:sa=mosc,moa,seq_cst,acquire,seq_cst:ss=mosc,mosc,seq_cst,acquire,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// %FILECHECK% PREFIX_COMBINE nvvm,byte
// %FILECHECK% PREFIX_COMBINE nvvm,byte,block
// %FILECHECK% PREFIX_COMBINE nvvm,byte,non_block
// %FILECHECK% PREFIX_COMBINE ptx,block
// %FILECHECK% PREFIX_COMBINE ptx,non_block
// %FILECHECK% PREFIX_COMBINE ptx,seq_cst
// %FILECHECK% PREFIX_COMBINE ptx,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE ptx,release
// %FILECHECK% PREFIX_COMBINE sm75,nvvm,halfword
// %FILECHECK% PREFIX_COMBINE sm75,nvvm,halfword,seq_cst
// %FILECHECK% PREFIX_COMBINE sm75,nvvm,halfword,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE sm75,nvvm,halfword,release
// %FILECHECK% PREFIX_COMBINE sm75,nvvm,byte,seq_cst
// %FILECHECK% PREFIX_COMBINE sm75,nvvm,byte,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE sm75,nvvm,byte,release
// %FILECHECK% PREFIX_COMBINE sm80,nvvm,halfword
// %FILECHECK% PREFIX_COMBINE sm80,nvvm,halfword,seq_cst
// %FILECHECK% PREFIX_COMBINE sm80,nvvm,halfword,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE sm80,nvvm,halfword,release
// %FILECHECK% PREFIX_COMBINE sm80,nvvm,byte,seq_cst
// %FILECHECK% PREFIX_COMBINE sm80,nvvm,byte,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE sm80,nvvm,byte,release
// %FILECHECK% PREFIX_COMBINE sm90,nvvm,halfword
// %FILECHECK% PREFIX_COMBINE sm90,nvvm,halfword,seq_cst
// %FILECHECK% PREFIX_COMBINE sm90,nvvm,halfword,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE sm90,nvvm,halfword,release
// %FILECHECK% PREFIX_COMBINE sm90,nvvm,byte,seq_cst
// %FILECHECK% PREFIX_COMBINE sm90,nvvm,byte,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE sm90,nvvm,byte,release
// %FILECHECK% PREFIX_COMBINE sm100,nvvm,halfword
// %FILECHECK% PREFIX_COMBINE sm100,nvvm,halfword,block
// %FILECHECK% PREFIX_COMBINE sm100,nvvm,halfword,non_block
// %FILECHECK% PREFIX_COMBINE sm100,nvvm,seq_cst
// %FILECHECK% PREFIX_COMBINE sm100,nvvm,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE sm100,nvvm,release
// %FILECHECK% PREFIX_COMBINE sm120,nvvm,halfword
// %FILECHECK% PREFIX_COMBINE sm120,nvvm,halfword,block
// %FILECHECK% PREFIX_COMBINE sm120,nvvm,halfword,non_block
// %FILECHECK% PREFIX_COMBINE sm120,nvvm,seq_cst
// %FILECHECK% PREFIX_COMBINE sm120,nvvm,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE sm120,nvvm,release
// clang-format on

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "atomic_codegen_helpers.h"

extern "C" __device__ bool atomic_codegen_test(cuda::atomic_ref<TYPE, SCOPE>& atom, TYPE& expected, TYPE desired)
{
  return atom.CAS(expected, desired, SUCCESS_ORDER, FAILURE_ORDER);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; SM100_NVVM_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SM120_NVVM_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SM100_NVVM_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SM120_NVVM_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SM100_NVVM_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; SM120_NVVM_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; SM75_NVVM_HALFWORD_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SM80_NVVM_HALFWORD_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SM90_NVVM_HALFWORD_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SM90_NVVM_HALFWORD_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; PTX-DAG: {{.*}}LOP3.LUT [[ALIGNED_ADDR:R[0-9]+]], [[ATOM_ADDR]]{{(\.reuse)?}}, 0xfffffffc, {{.*}}
; NVVM_BYTE-DAG: {{.*}}LOP3.LUT [[ALIGNED_ADDR:R[0-9]+]], [[ATOM_ADDR]]{{(\.reuse)?}}, 0xfffffffc, {{.*}}
; SM75_NVVM_HALFWORD-DAG: {{.*}}LOP3.LUT [[ALIGNED_ADDR:R[0-9]+]], [[ATOM_ADDR]]{{(\.reuse)?}}, 0xfffffffd, {{.*}}
; SM80_NVVM_HALFWORD-DAG: {{.*}}LOP3.LUT [[ALIGNED_ADDR:R[0-9]+]], [[ATOM_ADDR]]{{(\.reuse)?}}, 0xfffffffd, {{.*}}
; SM90_NVVM_HALFWORD-DAG: {{.*}}LOP3.LUT [[ALIGNED_ADDR:R[0-9]+]], [[ATOM_ADDR]]{{(\.reuse)?}}, 0xfffffffd, {{.*}}
; SM100_NVVM_HALFWORD-DAG: {{.*}}LOP3.LUT [[ALIGNED_ADDR:R[0-9]+]], [[ATOM_ADDR]]{{(\.reuse)?}}, 0xfffffffc, {{.*}}
; SM120_NVVM_HALFWORD-DAG: {{.*}}LOP3.LUT [[ALIGNED_ADDR:R[0-9]+]], [[ATOM_ADDR]]{{(\.reuse)?}}, 0xfffffffc, {{.*}}
; SM75_NVVM_HALFWORD_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; SM80_NVVM_HALFWORD_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; SM75_NVVM_HALFWORD_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SM80_NVVM_HALFWORD_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; PTX_BLOCK: {{.*}}LD.E.STRONG.{{CTA|SM}} {{R[0-9]+}}, {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; PTX_NON_BLOCK: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] {{R[0-9]+}}, {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; NVVM_BYTE_BLOCK: {{.*}}LD.E.STRONG.{{CTA|SM}} {{R[0-9]+}}, {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; NVVM_BYTE_NON_BLOCK: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] {{R[0-9]+}}, {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; SM75_NVVM_HALFWORD: {{.*}}LD.E.SYS {{R[0-9]+}}, {{.*\[}}[[ALIGNED_ADDR]]{{\].*}}
; SM80_NVVM_HALFWORD: {{.*}}LD.E {{R[0-9]+}}, {{.*\[}}[[ALIGNED_ADDR]]{{\].*}}
; SM90_NVVM_HALFWORD: {{.*}}LD.E {{R[0-9]+}}, {{.*\[}}[[ALIGNED_ADDR]]{{\].*}}
; SM100_NVVM_HALFWORD_BLOCK: {{.*}}LD.E.STRONG.{{CTA|SM}} {{R[0-9]+}}, {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; SM100_NVVM_HALFWORD_NON_BLOCK: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] {{R[0-9]+}}, {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; SM120_NVVM_HALFWORD_BLOCK: {{.*}}LD.E.STRONG.{{CTA|SM}} {{R[0-9]+}}, {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; SM120_NVVM_HALFWORD_NON_BLOCK: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] {{R[0-9]+}}, {{.*\[}}[[ALIGNED_ADDR]]{{(\.64)?\].*}}
; PTX_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SM75_NVVM_BYTE_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SM80_NVVM_BYTE_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SM90_NVVM_BYTE_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; SM90_NVVM_HALFWORD_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; PTX_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SM75_NVVM_BYTE_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SM80_NVVM_BYTE_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; SM90_NVVM_BYTE_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; PTX_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; SM75_NVVM_BYTE_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; SM80_NVVM_BYTE_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; SM90_NVVM_BYTE_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; BLOCK: {{.*}}ATOM.E.CAS.STRONG.{{CTA|SM}}{{.*\[}}[[ALIGNED_ADDR]]{{\].*}}
; NON_BLOCK: {{.*}}ATOM.E.CAS.STRONG.[[SASS_SCOPE]]{{.*\[}}[[ALIGNED_ADDR]]{{\].*}}
; NON_BLOCK_ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
