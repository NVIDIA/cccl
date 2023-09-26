//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      printf("\nNVRTC ERROR: %s failed with error %s\n",          \
        #x, nvrtcGetErrorString(result));                         \
      exit(1);                                                    \
    }                                                             \
  } while(0)

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      printf("\nCUDA ERROR: %s failed with error %s\n", #x, msg); \
      exit(1);                                                    \
    }                                                             \
  } while(0)

#define CUDA_API_CALL(x) \
  do {                                                            \
    cudaError_t err = x;                                          \
    if (err != cudaSuccess)                                       \
    {                                                             \
      printf("\nCUDA ERROR: %s: %s\n",                            \
        cudaGetErrorName(err), cudaGetErrorString(err));          \
      exit(1);                                                    \
    }                                                             \
  } while(0)
