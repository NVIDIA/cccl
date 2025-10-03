// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdio>

#include <cuda.h>
#include <nvrtc.h>

static inline void check_nvrtc_result(nvrtcResult res, const char* lineinfo, const char* line)
{
  if (res != NVRTC_SUCCESS)
  {
    printf("\n%s: NVRTC ERROR - %s returned %s\n", lineinfo, line, nvrtcGetErrorString(res));
    exit(1);
  }
}

static inline void check_cuda_driver_result(CUresult res, const char* lineinfo, const char* line)
{
  if (res != CUDA_SUCCESS)
  {
    const char* msg;
    cuGetErrorName(res, &msg);
    printf("\n%s: CUDA DRIVER ERROR - %s returned %s\n", lineinfo, line, msg);
    exit(1);
  }
}

#define SAFE_CALL_2(handler, file, line, ...) handler(__VA_ARGS__, file ":" #line, #__VA_ARGS__);
#define SAFE_CALL_1(handler, file, line, ...) SAFE_CALL_2(handler, file, line, __VA_ARGS__)

// Detect errors when calling NVRTC functions and print the error message.
#define NVRTC_SAFE_CALL(...) SAFE_CALL_1(check_nvrtc_result, __FILE__, __LINE__, __VA_ARGS__)
// Detect errors when calling CUDA Driver functions and print the error message.
#define CUDA_SAFE_CALL(...) SAFE_CALL_1(check_cuda_driver_result, __FILE__, __LINE__, __VA_ARGS__)
