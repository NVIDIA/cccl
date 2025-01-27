/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cuda/__cccl_config>

#ifndef TEST_HALF_T
#  if defined(_CCCL_HAS_NVFP16) && defined(_LIBCUDACXX_HAS_NVFP16)
#    define TEST_HALF_T() 1
#  else // defined(_CCCL_HAS_NVFP16) && defined(_LIBCUDACXX_HAS_NVFP16)
#    define TEST_HALF_T() 0
#  endif // defined(_CCCL_HAS_NVFP16) && defined(_LIBCUDACXX_HAS_NVFP16)
#endif // TEST_HALF_T

#ifndef TEST_BF_T
#  if defined(_CCCL_HAS_NVBF16) && defined(_LIBCUDACXX_HAS_NVBF16)
#    define TEST_BF_T() 1
#  else // defined(_CCCL_HAS_NVBF16) && defined(_LIBCUDACXX_HAS_NVBF16)
#    define TEST_BF_T() 0
#  endif // defined(_CCCL_HAS_NVBF16) && defined(_LIBCUDACXX_HAS_NVBF16)
#endif // TEST_BF_T

#if TEST_HALF_T()
#  include <cuda_fp16.h>

#  include <c2h/half.cuh>
#endif // TEST_HALF_T()

#if TEST_BF_T()
#  include <cuda_bf16.h>

#  include <c2h/bfloat16.cuh>
#endif // TEST_BF_T()
