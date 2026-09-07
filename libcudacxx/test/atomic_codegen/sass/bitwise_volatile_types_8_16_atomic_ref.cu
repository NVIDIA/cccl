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
// %PARAM% TYPE type int8_t:uint8_t:int16_t:uint16_t
// %PARAM% OP,SASS_OP,FILECHECK_PREFIX_BITWISE_OP op and=fetch_and,AND,and:or=fetch_or,OR,or_xor:xor=fetch_xor,XOR,or_xor
// %PARAM% ORDER,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,non_seq_cst,no_acquire,no_membar:acquire=moa,non_seq_cst,acquire,no_membar:release=more,non_seq_cst,no_acquire,release:acq_rel=moar,non_seq_cst,acquire,release:seq_cst=mosc,seq_cst,acquire,seq_cst
// %FILECHECK% PREFIX_COMBINE cas_subword_bitwise,block
// %FILECHECK% PREFIX_COMBINE cas_subword_bitwise,non_block
// %FILECHECK% PREFIX_COMBINE cas_subword_bitwise,non_seq_cst
// %FILECHECK% PREFIX_COMBINE cas_subword_bitwise,seq_cst
// %FILECHECK% PREFIX_COMBINE cas_subword_bitwise,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE cas_subword_bitwise,release
// %FILECHECK% PREFIX_COMBINE cas_subword_bitwise,no_membar
// %FILECHECK% PREFIX_COMBINE cas_subword_bitwise,non_block,acquire
// %FILECHECK% PREFIX_COMBINE cas_subword_bitwise,no_acquire
// %FILECHECK% PREFIX_COMBINE cas_subword_bitwise,cas_compare_old_expected
// %FILECHECK% PREFIX_COMBINE cas_subword_bitwise,cas_compare_expected_old
// %FILECHECK% PREFIX_COMBINE native_subword_bitwise,block
// %FILECHECK% PREFIX_COMBINE native_subword_bitwise,non_block
// %FILECHECK% PREFIX_COMBINE native_subword_bitwise,non_seq_cst
// %FILECHECK% PREFIX_COMBINE native_subword_bitwise,seq_cst
// %FILECHECK% PREFIX_COMBINE native_subword_bitwise,non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE native_subword_bitwise,release
// %FILECHECK% PREFIX_COMBINE native_subword_bitwise,no_membar
// %FILECHECK% PREFIX_COMBINE native_subword_bitwise,non_block,acquire
// %FILECHECK% PREFIX_COMBINE native_subword_bitwise,no_acquire
// %FILECHECK% PREFIX_COMBINE native_subword_bitwise,and
// %FILECHECK% PREFIX_COMBINE native_subword_bitwise,or_xor
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ auto atomic_codegen_test(cuda::atomic_ref<volatile TYPE, SCOPE>& atom, TYPE value)
{
  return atom.OP(value, ORDER);
}

// The compiler may fuse the bitwise operation with subword packing, so avoid
// fixing a particular LOP3 truth table.
// TODO: Improve PTX codegen for SM100+ to use the native widened and masked
// subword bitwise operations too.
// clang-format off
/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX: {{.*}}LD.E.64{{(\.SYS)?}} [[ATOM_ADDR:R[0-9]+]], {{.*}}
; CAS_SUBWORD_BITWISE-NOT: {{.*}}ATOM.E.[[SASS_OP]]{{.*}}
; CAS_SUBWORD_BITWISE_BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; CAS_SUBWORD_BITWISE_NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; CAS_SUBWORD_BITWISE: {{.*}}LOP3.LUT [[CAS_ALIGNED_ADDR:R[0-9]+]], [[ATOM_ADDR]]{{(\.reuse)?}}, 0xfffffffc, {{.*}}
; CAS_SUBWORD_BITWISE_BLOCK: {{.*}}LD.E.STRONG.{{CTA|SM}} [[EXPECTED:R[0-9]+]], {{.*\[}}[[CAS_ALIGNED_ADDR]]{{(\.64)?\].*}}
; CAS_SUBWORD_BITWISE_NON_BLOCK: {{.*}}LD.E.STRONG.[[SASS_SCOPE]] [[EXPECTED:R[0-9]+]], {{.*\[}}[[CAS_ALIGNED_ADDR]]{{(\.64)?\].*}}
; CAS_SUBWORD_BITWISE: {{.*}}LOP3.LUT {{.*}}
; CAS_SUBWORD_BITWISE_RELEASE: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; CAS_SUBWORD_BITWISE_SEQ_CST: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; CAS_SUBWORD_BITWISE_NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; CAS_SUBWORD_BITWISE_BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; CAS_SUBWORD_BITWISE_NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; CAS_SUBWORD_BITWISE_NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; CAS_SUBWORD_BITWISE-NOT: {{.*}}ATOM.E.[[SASS_OP]]{{.*}}
; CAS_SUBWORD_BITWISE_BLOCK: {{.*}}ATOM.E.CAS.STRONG.{{CTA|SM}} PT, [[OLD:R[0-9]+]], {{\[}}[[CAS_ALIGNED_ADDR]]{{\]}}, [[EXPECTED]], {{R[0-9]+}}{{.*}}
; CAS_SUBWORD_BITWISE_NON_BLOCK: {{.*}}ATOM.E.CAS.STRONG.[[SASS_SCOPE]] PT, [[OLD:R[0-9]+]], {{\[}}[[CAS_ALIGNED_ADDR]]{{\]}}, [[EXPECTED]], {{R[0-9]+}}{{.*}}
; CAS_SUBWORD_BITWISE_BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; CAS_SUBWORD_BITWISE_NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; CAS_SUBWORD_BITWISE_CAS_COMPARE_OLD_EXPECTED-DAG: {{.*}}ISETP.NE{{.*}} [[OLD]]{{(\.reuse)?}}, [[EXPECTED]]{{(\.reuse)?}}, {{.*}}
; CAS_SUBWORD_BITWISE_CAS_COMPARE_EXPECTED_OLD-DAG: {{.*}}ISETP.NE{{.*}} [[EXPECTED]]{{(\.reuse)?}}, [[OLD]]{{(\.reuse)?}}, {{.*}}
; CAS_SUBWORD_BITWISE_NON_BLOCK_ACQUIRE-DAG: {{.*}}CCTL.IVALL{{.*}}
; CAS_SUBWORD_BITWISE_BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; CAS_SUBWORD_BITWISE_NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; CAS_SUBWORD_BITWISE-NOT: {{.*}}ATOM.E.[[SASS_OP]]{{.*}}
; CAS_SUBWORD_BITWISE-NOT: {{.*}}ATOM.E.CAS{{.*}}
; CAS_SUBWORD_BITWISE: {{.*}}RET.ABS.NODEC{{.*}}

; NATIVE_SUBWORD_BITWISE-NOT: {{.*}}ATOM.E.CAS{{.*}}
; NATIVE_SUBWORD_BITWISE_BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NATIVE_SUBWORD_BITWISE_NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; NATIVE_SUBWORD_BITWISE_RELEASE-DAG: {{.*}}MEMBAR.ALL.[[SASS_SCOPE]]{{.*}}
; NATIVE_SUBWORD_BITWISE_SEQ_CST-DAG: {{.*}}MEMBAR.SC.[[SASS_SCOPE]]{{.*}}
; NATIVE_SUBWORD_BITWISE_NON_BLOCK_SEQ_CST-DAG: {{.*}}CCTL.IVALL{{.*}}
; NATIVE_SUBWORD_BITWISE_NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; NATIVE_SUBWORD_BITWISE-DAG: {{.*}}LOP3.LUT [[NATIVE_ALIGNED_ADDR:R[0-9]+]], [[ATOM_ADDR]]{{(\.reuse)?}}, 0xfffffffc, {{.*}}
; NATIVE_SUBWORD_BITWISE-DAG: {{.*}}LOP3.LUT [[SHIFT:R[0-9]+]], {{.*}}, 0x18, {{.*}}
; NATIVE_SUBWORD_BITWISE_AND-DAG: {{.*}}SHF.L.U32 [[SHIFTED_OPERAND:R[0-9]+]], {{.*}}, [[SHIFT]]{{(\.reuse)?}}, RZ{{.*}}
; NATIVE_SUBWORD_BITWISE_AND-DAG: {{.*}}LOP3.LUT [[NATIVE_OPERAND:R[0-9]+]], [[SHIFTED_OPERAND]], {{.*}}
; NATIVE_SUBWORD_BITWISE_OR_XOR-DAG: {{.*}}SHF.L.U32 [[NATIVE_OPERAND:R[0-9]+]], {{.*}}, [[SHIFT]]{{(\.reuse)?}}, RZ{{.*}}
; NATIVE_SUBWORD_BITWISE-NOT: {{.*}}ATOM.E.CAS{{.*}}
; NATIVE_SUBWORD_BITWISE_BLOCK: {{.*}}ATOM.E.[[SASS_OP]].STRONG.{{CTA|SM}} PT, [[NATIVE_OLD:R[0-9]+]], {{.*\[}}[[NATIVE_ALIGNED_ADDR]]{{(\.64)?\].*}}, [[NATIVE_OPERAND]]{{.*}}
; NATIVE_SUBWORD_BITWISE_NON_BLOCK: {{.*}}ATOM.E.[[SASS_OP]].STRONG.[[SASS_SCOPE]] PT, [[NATIVE_OLD:R[0-9]+]], {{.*\[}}[[NATIVE_ALIGNED_ADDR]]{{(\.64)?\].*}}, [[NATIVE_OPERAND]]{{.*}}
; NATIVE_SUBWORD_BITWISE_BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NATIVE_SUBWORD_BITWISE_NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; NATIVE_SUBWORD_BITWISE_NON_BLOCK_ACQUIRE-DAG: {{.*}}CCTL.IVALL{{.*}}
; NATIVE_SUBWORD_BITWISE-DAG: {{.*}}SHF.R.U32.HI {{R[0-9]+}}, RZ, [[SHIFT]]{{(\.reuse)?}}, [[NATIVE_OLD]]{{.*}}
; NATIVE_SUBWORD_BITWISE-NOT: {{.*}}ATOM.E.[[SASS_OP]]{{.*}}
; NATIVE_SUBWORD_BITWISE-NOT: {{.*}}ATOM.E.CAS{{.*}}
; NATIVE_SUBWORD_BITWISE: {{.*}}RET.ABS.NODEC{{.*}}

*/
// clang-format on
