//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#if defined(__CUDACC__) && !defined(_NVHPC_CUDA)
#  define CUDASTF_NO_DEVICE_STACK
#elif defined(__CUDACC__) && defined(_NVHPC_CUDA)
#  define CUDASTF_NO_DEVICE_STACK _Pragma("diag_suppress no_device_stack")
#else
#  define CUDASTF_NO_DEVICE_STACK
#endif
