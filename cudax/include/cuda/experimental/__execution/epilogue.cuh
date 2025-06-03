//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// IMPORTANT: This file intentionally lacks a header guard.

#if !defined(_CUDAX_ASYNC_PROLOGUE_INCLUDED)
#  error epilogue.cuh included without a prior inclusion of prologue.cuh
#endif

#undef _CCCL_IMMOVABLE_OPSTATE
#undef _CUDAX_ASYNC_PROLOGUE_INCLUDED

#if _CCCL_CUDA_COMPILER(NVHPC)
_CCCL_NV_DIAG_DEFAULT(cuda_compile)
#endif // _CCCL_CUDA_COMPILER(NVHPC)

_CCCL_DIAG_POP
#include <cuda/std/__cccl/epilogue.h>
