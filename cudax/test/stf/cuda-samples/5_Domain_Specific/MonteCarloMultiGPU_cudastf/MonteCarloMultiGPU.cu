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

/*
 * This sample evaluates fair call price for a
 * given set of European options using Monte Carlo approach.
 * See supplied whitepaper for more explanations.
 */

#include "MonteCarlo_gold.cu"
#include "MonteCarlo_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
float randFloat(float low, float high)
{
  float t = (float) rand() / (float) RAND_MAX;
  return (1.0f - t) * low + t * high;
}

/// Utility function to tweak problem size for small GPUs
int adjustProblemSize(int GPU_N, int default_nOptions)
{
  int nOptions = default_nOptions;

  for (int i = 0; i < GPU_N; i++)
  {
    cudaDeviceProp deviceProp;
    cuda_safe_call(cudaGetDeviceProperties(&deviceProp, i));
    int cudaCores = 80;

    if (cudaCores <= 32)
    {
      nOptions = (nOptions < cudaCores / 2 ? nOptions : cudaCores / 2);
    }
  }

  return nOptions;
}

int adjustGridSize(int GPUIndex, int defaultGridSize)
{
  cudaDeviceProp deviceProp;
  cuda_safe_call(cudaGetDeviceProperties(&deviceProp, GPUIndex));
  int maxGridSize = deviceProp.multiProcessorCount * 40;
  return ((defaultGridSize > maxGridSize) ? maxGridSize : defaultGridSize);
}

///////////////////////////////////////////////////////////////////////////////
// CPU reference functions
///////////////////////////////////////////////////////////////////////////////
extern "C" void MonteCarloCPU(TOptionValue& callValue, TOptionData optionData, float* h_Random, int pathN);

// Black-Scholes formula for call options
extern "C" void BlackScholesCall(float& CallResult, TOptionData optionData);

////////////////////////////////////////////////////////////////////////////////
// Single-threaded multi-GPU solver using STF
////////////////////////////////////////////////////////////////////////////////
static void multiSolver(TOptionPlan* plan, int nPlans)
{
  stream_ctx ctx;

  for (int i = 0; i < nPlans; i++)
  {
    cuda_safe_call(cudaSetDevice(plan[i].device));

    initMonteCarloGPU(ctx, &plan[i]);
    MonteCarloGPU(ctx, &plan[i]);
    closeMonteCarloGPU(ctx, &plan[i]);
  }

  ctx.finalize();
}

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  int GPU_N;
  cuda_safe_call(cudaGetDeviceCount(&GPU_N));
  int nOptions = 8 * 1024;

  nOptions = adjustProblemSize(GPU_N, nOptions);

  int OPT_N  = nOptions * GPU_N;
  int PATH_N = 262144;

  // Input data array
  TOptionData* optionData = new TOptionData[OPT_N];
  // Final GPU MC results
  TOptionValue* callValueGPU = new TOptionValue[OPT_N];
  //"Theoretical" call values by Black-Scholes formula
  float* callValueBS = new float[OPT_N];
  // Solver config
  TOptionPlan* optionSolver = new TOptionPlan[GPU_N];

  int i;
  double delta, ref, sumDelta, sumRef, sumReserve;

  srand(123);

  for (i = 0; i < OPT_N; i++)
  {
    optionData[i].S            = randFloat(5.0f, 50.0f);
    optionData[i].X            = randFloat(10.0f, 25.0f);
    optionData[i].T            = randFloat(1.0f, 5.0f);
    optionData[i].R            = 0.06f;
    optionData[i].V            = 0.10f;
    callValueGPU[i].Expected   = -1.0f;
    callValueGPU[i].Confidence = -1.0f;
  }

  // Get option count for each GPU
  for (i = 0; i < GPU_N; i++)
  {
    optionSolver[i].optionCount = OPT_N / GPU_N;
  }

  // Take into account cases with "odd" option counts
  for (i = 0; i < (OPT_N % GPU_N); i++)
  {
    optionSolver[i].optionCount++;
  }

  // Assign GPU option ranges
  int gpuBase = 0;

  for (i = 0; i < GPU_N; i++)
  {
    optionSolver[i].device     = i;
    optionSolver[i].optionData = optionData + gpuBase;
    optionSolver[i].callValue  = callValueGPU + gpuBase;
    optionSolver[i].pathN      = PATH_N;
    optionSolver[i].gridSize   = adjustGridSize(optionSolver[i].device, optionSolver[i].optionCount);
    gpuBase += optionSolver[i].optionCount;
  }

  multiSolver(optionSolver, GPU_N);

  // Compare Monte Carlo and Black-Scholes results
  sumDelta   = 0;
  sumRef     = 0;
  sumReserve = 0;

  for (i = 0; i < OPT_N; i++)
  {
    BlackScholesCall(callValueBS[i], optionData[i]);
    delta = fabs(callValueBS[i] - callValueGPU[i].Expected);
    ref   = callValueBS[i];
    sumDelta += delta;
    sumRef += fabs(ref);

    if (delta > 1e-6)
    {
      sumReserve += callValueGPU[i].Confidence / delta;
    }
  }

  sumReserve /= OPT_N;

  delete[] optionSolver;
  delete[] callValueBS;
  delete[] callValueGPU;
  delete[] optionData;

  if (sumReserve <= 1.0f)
  {
    printf("Test failed!\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
