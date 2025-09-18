/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/par.h>

// define these entities here for the purpose of Doxygenating them
// they are actually defined elsewhere
#if 0
THRUST_NAMESPACE_BEGIN
namespace cuda
{

/*! \addtogroup execution_policies
 *  \{
 */

/*! \p thrust::cuda::par is the parallel execution policy associated with
 *  Thrust's CUDA device backend.
 *
 *  Instead of relying on implicit algorithm dispatch through iterator system tags,
 *  users may directly target Thrust's CUDA backend by providing \p thrust::cuda::par
 *  as an algorithm parameter. The policy can be attached to a specific CUDA stream
 *  using the `.on(stream)` helper.
 *
 *  The type of \p thrust::cuda::par is implementation-defined.
 */
static const unspecified par;

/*! \p thrust::cuda::par_nosync is a parallel execution policy targeting
 *  Thrust's CUDA device backend that allows algorithms to elide optional
 *  stream synchronizations.
 *
 *  Similar to \p thrust::cuda::par, it allows execution of Thrust algorithms in
 *  a specific CUDA stream via `.on(stream)`. In addition, \p thrust::cuda::par_nosync
 *  indicates that an algorithm is free to avoid any synchronization of the associated
 *  stream that is not strictly required for correctness. Algorithms may also return
 *  before the corresponding kernels are completed (similar to asynchronous kernel launches).
 *  The user must perform explicit synchronization when necessary.
 *
 *  Example:
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/for_each.h>
 *  #include <thrust/execution_policy.h>
 *
 *  struct IncFunctor{
 *    __host__ __device__
 *    void operator()(std::size_t& x){ x = x + 1; };
 *  };
 *
 *  cudaStream_t stream;
 *  cudaStreamCreate(&stream);
 *  auto nosync = thrust::cuda::par_nosync.on(stream);
 *
 *  thrust::for_each(nosync, vec.begin(), vec.end(), IncFunctor{});
 *  // ... do other host work ...
 *  cudaStreamSynchronize(stream);
 *  cudaStreamDestroy(stream);
 *  \endcode
 */
static const unspecified par_nosync;

/*! \}
 */

} // end cuda
THRUST_NAMESPACE_END
#endif
