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

////////////////////////////////////////////////////////////////////////////////
// Global types
////////////////////////////////////////////////////////////////////////////////
#include "MonteCarlo_reduction.cuh"

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError((msg), __FILE__, __LINE__)

namespace cg = cooperative_groups;

inline void __getLastCudaError(const char* errorMessage, const char* file, const int line)
{
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err)
  {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error:"
            " %s : (%d) %s.\n",
            file,
            line,
            errorMessage,
            static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Internal GPU-side data structures
////////////////////////////////////////////////////////////////////////////////
#define MAX_OPTIONS (1024 * 1024)

////////////////////////////////////////////////////////////////////////////////
// Overloaded shortcut payoff functions for different precision modes
////////////////////////////////////////////////////////////////////////////////
__device__ inline float endCallValue(float S, float X, float r, float MuByT, float VBySqrtT)
{
  float callValue = S * __expf(MuByT + VBySqrtT * r) - X;
  return (callValue > 0.0F) ? callValue : 0.0F;
}

__device__ inline double endCallValue(double S, double X, double r, double MuByT, double VBySqrtT)
{
  double callValue = S * exp(MuByT + VBySqrtT * r) - X;
  return (callValue > 0.0) ? callValue : 0.0;
}

#define THREAD_N 256

////////////////////////////////////////////////////////////////////////////////
// This kernel computes the integral over all paths using a single thread block
// per option. It is fastest when the number of thread blocks times the work per
// block is high enough to keep the GPU busy.
////////////////////////////////////////////////////////////////////////////////
static __global__ void MonteCarloOneBlockPerOption(
  curandState* __restrict rngStates,
  const __TOptionData* __restrict d_OptionData,
  __TOptionValue* __restrict d_CallValue,
  int pathN,
  int optionN)
{
  // Handle to thread block group
  cg::thread_block cta             = cg::this_thread_block();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  const int SUM_N = THREAD_N;
  __shared__ real s_SumCall[SUM_N];
  __shared__ real s_Sum2Call[SUM_N];

  // determine global thread id
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Copy random number state to local memory for efficiency
  curandState localState = rngStates[tid];
  for (int optionIndex = blockIdx.x; optionIndex < optionN; optionIndex += gridDim.x)
  {
    const real S        = d_OptionData[optionIndex].S;
    const real X        = d_OptionData[optionIndex].X;
    const real MuByT    = d_OptionData[optionIndex].MuByT;
    const real VBySqrtT = d_OptionData[optionIndex].VBySqrtT;

    // Cycle through the entire samples array:
    // derive end stock price for each path
    // accumulate partial integrals into intermediate shared memory buffer
    for (int iSum = threadIdx.x; iSum < SUM_N; iSum += blockDim.x)
    {
      __TOptionValue sumCall = {0, 0};

      _CCCL_PRAGMA_UNROLL(8)
      for (int i = iSum; i < pathN; i += SUM_N)
      {
        real r         = curand_normal(&localState);
        real callValue = endCallValue(S, X, r, MuByT, VBySqrtT);
        sumCall.Expected += callValue;
        sumCall.Confidence += callValue * callValue;
      }

      s_SumCall[iSum]  = sumCall.Expected;
      s_Sum2Call[iSum] = sumCall.Confidence;
    }

    // Reduce shared memory accumulators
    // and write final result to global memory
    cg::sync(cta);
    sumReduce<real, SUM_N, THREAD_N>(s_SumCall, s_Sum2Call, cta, tile32, &d_CallValue[optionIndex]);
  }
}

static __global__ void rngSetupStates(curandState* rngState, int device_id)
{
  // determine global thread id
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // Each threadblock gets different seed,
  // Threads within a threadblock get different sequence numbers
  curand_init(blockIdx.x + gridDim.x * device_id, threadIdx.x, 0, &rngState[tid]);
}

////////////////////////////////////////////////////////////////////////////////
// Host-side interface to GPU Monte Carlo
////////////////////////////////////////////////////////////////////////////////

template <typename Ctx>
void initMonteCarloGPU(Ctx& ctx, TOptionPlan* plan)
{
  plan->h_OptionData = new __TOptionData[plan->optionCount];
  plan->h_CallValue  = new __TOptionValue[plan->optionCount];
  cuda_safe_call(
    cudaHostRegister(plan->h_OptionData, plan->optionCount * sizeof(__TOptionData), cudaHostRegisterPortable));
  cuda_safe_call(
    cudaHostRegister(plan->h_CallValue, plan->optionCount * sizeof(__TOptionValue), cudaHostRegisterPortable));

  // Register this vector
  plan->preproc_optionData_handle = ctx.logical_data((__TOptionData*) plan->h_OptionData, plan->optionCount);
  plan->callValue_handle          = ctx.logical_data((__TOptionValue*) plan->h_CallValue, plan->optionCount);
  plan->rngStates_handle          = ctx.logical_data(shape_of<slice<curandState>>(plan->gridSize * THREAD_N));

  plan->preproc_optionData_handle.set_symbol("preproc_optionData");
  plan->callValue_handle.set_symbol("callValue");
  plan->rngStates_handle.set_symbol("rngStates");

  cuda_safe_call(cudaSetDevice(plan->device));

  // Allocate states for pseudo random number generators
  auto t = ctx.task(plan->rngStates_handle.write());
  t.set_symbol("rngSetupStates");
  t->*[&](cudaStream_t stream, auto rngStates) {
    // fprintf(stderr, "MEMSET on %zu\n", plan->gridSize * THREAD_N * sizeof(curandState));
    cuda_safe_call(
      cudaMemsetAsync(rngStates.data_handle(), 0, plan->gridSize * THREAD_N * sizeof(curandState), stream));
    getLastCudaError("cudaMemsetAsync failed.\n");

    // place each device pathN random numbers apart on the random number sequence
    // fprintf(stderr, "rngSetupStates => rngStates %p\n", rngStates.data_handle());
    rngSetupStates<<<plan->gridSize, THREAD_N, 0, stream>>>(rngStates.data_handle(), plan->device);
    getLastCudaError("rngSetupStates kernel failed.\n");
  };
}

// Compute statistics and deallocate internal device memory
template <typename Ctx>
void closeMonteCarloGPU(Ctx& ctx, TOptionPlan* plan)
{
  auto t = ctx.task(exec_place::host(), plan->callValue_handle.rw());
  t.set_symbol("compute_stats");
  t->*[&](cudaStream_t stream, auto h_CallValue) {
    cuda_safe_call(cudaStreamSynchronize(stream));
    for (int i = 0; i < plan->optionCount; i++)
    {
      const double RT    = plan->optionData[i].R * plan->optionData[i].T;
      const double sum   = h_CallValue.data_handle()[i].Expected;
      const double sum2  = h_CallValue.data_handle()[i].Confidence;
      const double pathN = plan->pathN;
      // Derive average from the total sum and discount by riskfree rate
      plan->callValue[i].Expected = (float) (exp(-RT) * sum / pathN);
      // Standard deviation
      double stdDev = sqrt((pathN * sum2 - sum * sum) / (pathN * (pathN - 1)));
      // Confidence width; in 95% of all cases theoretical value lies within these
      // borders
      plan->callValue[i].Confidence = (float) (exp(-RT) * 1.96 * stdDev / sqrt(pathN));
    }
  };
}

// Main computations
template <typename Ctx>
void MonteCarloGPU(Ctx& ctx, TOptionPlan* plan, cudaStream_t /*unused*/ = 0)
{
  if (plan->optionCount <= 0 || plan->optionCount > MAX_OPTIONS)
  {
    printf("MonteCarloGPU(): bad option count.\n");
    return;
  }

  // Preprocess computations on the host
  auto t_host = ctx.task(exec_place::host(), plan->preproc_optionData_handle.rw());
  t_host.set_symbol("preprocess");
  t_host->*[&](cudaStream_t stream, auto h_preproc_OptionData) {
    cuda_safe_call(cudaStreamSynchronize(stream));

    for (int i = 0; i < plan->optionCount; i++)
    {
      const double T                                 = plan->optionData[i].T;
      const double R                                 = plan->optionData[i].R;
      const double V                                 = plan->optionData[i].V;
      const double MuByT                             = (R - 0.5 * V * V) * T;
      const double VBySqrtT                          = V * sqrt(T);
      h_preproc_OptionData.data_handle()[i].S        = (real) plan->optionData[i].S;
      h_preproc_OptionData.data_handle()[i].X        = (real) plan->optionData[i].X;
      h_preproc_OptionData.data_handle()[i].MuByT    = (real) MuByT;
      h_preproc_OptionData.data_handle()[i].VBySqrtT = (real) VBySqrtT;
    }
  };

  auto t =
    ctx.task(plan->preproc_optionData_handle.read(), plan->callValue_handle.write(), plan->rngStates_handle.rw());
  t.set_symbol("MonteCarloOneBlockPerOption");
  t->*[&](cudaStream_t stream, auto preproc_optionData, auto callValue_handle, auto rngStates) {
    MonteCarloOneBlockPerOption<<<plan->gridSize, THREAD_N, 0, stream>>>(
      rngStates.data_handle(),
      preproc_optionData.data_handle(),
      callValue_handle.data_handle(),
      plan->pathN,
      plan->optionCount);
    getLastCudaError("MonteCarloOneBlockPerOption() execution failed\n");
  };
}
