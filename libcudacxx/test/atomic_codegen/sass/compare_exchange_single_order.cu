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
// %PARAM% SINGLE_ORDER,OVERLOAD_KIND overload single=1,single:pair=0,pair
// %PARAM% CAS cas compare_exchange_weak:compare_exchange_strong
// %PARAM% SUCCESS_ORDER,FAILURE_ORDER,SASS_MEMBAR,FILECHECK_PREFIX_ORDER order relaxed=mor,mor,,no_membar:acquire=moa,moa,,no_membar:release=more,mor,ALL,membar:acq_rel=moar,moa,ALL,membar:seq_cst=mosc,mosc,SC,membar
// clang-format on

#include "atomic_codegen_helpers.h"

__device__ bool
atomic_compare_exchange(cuda::atomic_ref<int32_t, cuda::thread_scope_device>& atom, int32_t& expected, int32_t desired)
{
#if SINGLE_ORDER
  return atom.CAS(expected, desired, SUCCESS_ORDER);
#else // ^^^ SINGLE_ORDER ^^^ / vvv !SINGLE_ORDER vvv
  return atom.CAS(expected, desired, SUCCESS_ORDER, FAILURE_ORDER);
#endif // !SINGLE_ORDER
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*atomic_compare_exchange.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; MEMBAR: {{.*}}MEMBAR.[[SASS_MEMBAR]].GPU{{.*}}
; NO_MEMBAR-NOT: {{.*}}MEMBAR.{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}ATOM.E.CAS.STRONG.GPU{{.*}}
; SMXX-NOT: {{.*}}ATOM.E.EXCH{{.*}}
; SMXX: {{.*}}RET.ABS.NODEC{{.*}}

*/
