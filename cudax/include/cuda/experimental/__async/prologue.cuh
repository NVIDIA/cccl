//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include "config.cuh"

#if defined(_CUDAX_ASYNC_PROLOGUE_INCLUDED)
#  error multiple inclusion of prologue.cuh
#endif

#define _CUDAX_ASYNC_PROLOGUE_INCLUDED

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wsubobject-linkage")
