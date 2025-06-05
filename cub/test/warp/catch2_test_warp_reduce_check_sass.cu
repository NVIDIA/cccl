/***********************************************************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and the
 *       following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
 *       following disclaimer in the documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************************************************************/
// #define CCCL_CHECK_SASS

#if defined(CCCL_CHECK_SASS)
#  include <cub/warp/warp_reduce.cuh>

#  include <test_util.h>

#  include <c2h/catch2_test_helper.h>
#  include <c2h/extended_types.h>

__device__ uint64_t input[256];
__device__ uint64_t output[256];

template <typename T, typename ReductionOp>
__global__ void warp_reduce_kernel(ReductionOp reduction_op)
{
  using storage_t = typename cub::WarpReduce<T>::TempStorage;
  storage_t storage{};
  auto value      = *reinterpret_cast<T*>(&input + threadIdx.x);
  auto output_ptr = reinterpret_cast<T*>(&output + threadIdx.x);
  *output_ptr     = cub::WarpReduce<T>{storage}.Reduce(value, reduction_op);
}

/***********************************************************************************************************************
 * Types
 **********************************************************************************************************************/
// clang-format off

using arithmetic_type_list = c2h::type_list<
  int32_t, int64_t,
  float, double,
  cuda::std::complex<float>,
  cuda::std::complex<double>,
  ushort2
#  if TEST_HALF_T()
   , __half
   , __half2
#  endif // TEST_HALF_T()
#  if TEST_BF_T()
   , __nv_bfloat16
   , __nv_bfloat162
#  endif // TEST_BF_T()
>;


using bitwise_type_list = c2h::type_list<uint32_t, uint64_t>;

using min_max_type_list = c2h::type_list<
  int32_t, int64_t,
   ushort2,
  float, double
#  if TEST_HALF_T()
  , __half
   , __half2
#  endif // TEST_HALF_T()
#  if TEST_BF_T()
  , __nv_bfloat16
   , __nv_bfloat162
#  endif // TEST_BF_T()
  >;

// clang-format on
/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/

C2H_TEST("SASS WarpReduce::Sum", "[reduce][warp][predefined_op][full]", arithmetic_type_list)
{
  using T = c2h::get<0, TestType>;
  warp_reduce_kernel<T><<<1, 32>>>(cuda::std::plus<>{});
}

C2H_TEST("SASS WarpReduce::Bitwise", "[reduce][warp][predefined_op][full]", bitwise_type_list)
{
  using T = c2h::get<0, TestType>;
  warp_reduce_kernel<T><<<1, 32>>>(cuda::std::bit_and<>{});
}

C2H_TEST("SASS WarpReduce::Min/Max", "[reduce][warp][predefined_op][full]", min_max_type_list)
{
  using T = c2h::get<0, TestType>;
  warp_reduce_kernel<T><<<1, 32>>>(cuda::minimum<>{});
}

#else

#  include "c2h/catch2_test_helper.h"

C2H_TEST("WarpReduce Empty Test", "[reduce][thread][empty]") {}

#endif // defined(CCCL_CHECK_SASS)
