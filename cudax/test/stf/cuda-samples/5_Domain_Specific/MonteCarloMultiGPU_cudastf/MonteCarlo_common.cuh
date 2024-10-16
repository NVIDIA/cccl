//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef MONTECARLO_COMMON_H
#define MONTECARLO_COMMON_H
#include <cuda/experimental/stf.cuh>

#include "curand_kernel.h"
#include "realtype.cuh"

using namespace cuda::experimental::stf;

////////////////////////////////////////////////////////////////////////////////
// Global types
////////////////////////////////////////////////////////////////////////////////
typedef struct
{
  float S;
  float X;
  float T;
  float R;
  float V;
} TOptionData;

typedef struct
// #ifdef __CUDACC__
//__align__(8)
// #endif
{
  float Expected;
  float Confidence;
} TOptionValue;

// Preprocessed input option data
typedef struct
{
  real S;
  real X;
  real MuByT;
  real VBySqrtT;
} __TOptionData;

// GPU outputs before CPU postprocessing
typedef struct
{
  real Expected;
  real Confidence;
} __TOptionValue;

typedef struct
{
  // Device ID for multi-GPU version
  int device;
  // Option count for this plan
  int optionCount;

  // Host-side data source and result destination
  TOptionData* optionData;
  TOptionValue* callValue;
  logical_data<slice<__TOptionData>> preproc_optionData_handle;
  logical_data<slice<__TOptionValue>> callValue_handle;

  // Temporary Host-side pinned memory for async + faster data transfers
  __TOptionValue* h_CallValue;

  // Device- and host-side option data
  //    void* d_OptionData;
  void* h_OptionData;

  // Device-side option values
  //    void* d_CallValue;

  // Intermediate device-side buffers
  //    void* d_Buffer;

  // random number generator states
  curandState* rngStates;
  logical_data<slice<curandState>> rngStates_handle;

  // Pseudorandom samples count
  int pathN;

  // Time stamp
  float time;

  int gridSize;
} TOptionPlan;

// extern "C" void initMonteCarloGPU(TOptionPlan *plan);
// extern "C" void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream = 0);
// extern "C" void closeMonteCarloGPU(TOptionPlan *plan);

#endif
