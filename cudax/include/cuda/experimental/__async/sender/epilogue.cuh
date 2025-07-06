//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#if !defined(_CUDAX_ASYNC_PROLOGUE_INCLUDED)
#  __error epilogue.cuh included without a prior inclusion of prologue.cuh
#endif

#undef _CUDAX_ASYNC_PROLOGUE_INCLUDED

#if _CCCL_CUDA_COMPILER(NVHPC)
_CCCL_END_NV_DIAG_SUPPRESS()
#endif // _CCCL_CUDA_COMPILER(NVHPC)

_CCCL_DIAG_POP
