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
#    define CUDASTF_HOST_DEVICE __host__ __device__
#    define CUDASTF_DEVICE __device__
#    define CUDASTF_HOST __host__
#else
#    define CUDASTF_HOST_DEVICE
#    define CUDASTF_DEVICE
#    define CUDASTF_HOST
#endif

#if __CUDA_ARCH__ >= 700
#    define CUDASTF_GRID_CONSTANT __grid_constant__
#else
#    define CUDASTF_GRID_CONSTANT
#endif
