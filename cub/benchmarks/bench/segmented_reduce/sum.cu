/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1
// %RANGE% TUNE_S_THREADS_PER_WARP stpw 1:32:1
// %RANGE% TUNE_M_THREADS_PER_WARP mtpw 1:32:1
// %RANGE% TUNE_L_NOMINAL_4B_THREADS_PER_BLOCK ltpb 128:1024:32
// %RANGE% TUNE_S_NOMINAL_4B_ITEMS_PER_THREAD sipt 1:32:1
// %RANGE% TUNE_M_NOMINAL_4B_ITEMS_PER_THREAD mipt 1:32:1
// %RANGE% TUNE_L_NOMINAL_4B_ITEMS_PER_THREAD lipt 7:24:1

using value_types = all_types;
using op_t        = ::cuda::std::plus<>;
#include "base.cuh"
