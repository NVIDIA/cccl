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
// %PARAM% SINGLE_ORDER,OVERLOAD_KIND,FILECHECK_PREFIX_SCOPE overload single=1,single,non_block:pair=0,pair,non_block
// %PARAM% CAS cas compare_exchange_weak:compare_exchange_strong
// %PARAM% SUCCESS_ORDER,FAILURE_ORDER,SASS_MEMBAR,FILECHECK_PREFIX_SEQ_CST,FILECHECK_PREFIX_ACQUIRE,FILECHECK_PREFIX_ORDER order relaxed=mor,mor,,non_seq_cst,no_acquire,no_membar:acquire=moa,moa,,non_seq_cst,acquire,no_membar:release=more,mor,ALL,non_seq_cst,no_acquire,membar:acq_rel=moar,moa,ALL,non_seq_cst,acquire,membar:seq_cst=mosc,mosc,SC,seq_cst,acquire,membar
// %FILECHECK% PREFIX_COMBINE non_block,seq_cst
// %FILECHECK% PREFIX_COMBINE non_block,acquire
// clang-format on

#include "atomic_codegen_helpers.h"

extern "C" __device__ bool
atomic_codegen_test(cuda::atomic_ref<int32_t, cuda::thread_scope_device>& atom, int32_t& expected, int32_t desired)
{
#if SINGLE_ORDER
  return atom.CAS(expected, desired, SUCCESS_ORDER);
#else // ^^^ SINGLE_ORDER ^^^ / vvv !SINGLE_ORDER vvv
  return atom.CAS(expected, desired, SUCCESS_ORDER, FAILURE_ORDER);
#endif // !SINGLE_ORDER
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : atomic_codegen_test
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].GPU{{.*}}
; NON_BLOCK_SEQ_CST: {{.*}}CCTL.IVALL{{.*}}
; NON_SEQ_CST-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}ATOM.E.CAS.STRONG.GPU{{.*}}
; NON_BLOCK_ACQUIRE: {{.*}}CCTL.IVALL{{.*}}
; BLOCK-NOT: {{.*}}CCTL.IVALL{{.*}}
; NO_ACQUIRE-NOT: {{.*}}CCTL.IVALL{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
