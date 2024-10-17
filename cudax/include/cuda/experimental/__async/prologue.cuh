//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/std/detail/__config>

#if defined(_CUDAX_ASYNC_PROLOGUE_INCLUDED)
#  __error multiple inclusion of prologue.cuh
#endif

#define _CUDAX_ASYNC_PROLOGUE_INCLUDED

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wsubobject-linkage")
_CCCL_DIAG_SUPPRESS_MSVC(4848) // [[no_unique_address]] prior to C++20 as a vendor extension

_CCCL_DIAG_SUPPRESS_GCC("-Wmissing-braces")
_CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-braces")
_CCCL_DIAG_SUPPRESS_MSVC(5246) // missing braces around initializer
