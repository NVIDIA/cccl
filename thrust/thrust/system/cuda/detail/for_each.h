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

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#include <thrust/system/cuda/config.h>

#include <cub/device/device_for.cuh>

#include <thrust/system/cuda/detail/cdp_dispatch.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/system/cuda/detail/parallel_for.h>
#include <thrust/detail/function.h>
#include <thrust/distance.h>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub {

  // for_each_n
  _CCCL_EXEC_CHECK_DISABLE
  template <class Derived,
            class Input,
            class Size,
            class UnaryOp>
  Input THRUST_FUNCTION
  for_each_n(execution_policy<Derived> &policy,
             Input                      first,
             Size                       count,
             UnaryOp                    op)
  {
    THRUST_CDP_DISPATCH(
      (cudaStream_t stream = cuda_cub::stream(policy);
       cudaError_t  status = cub::DeviceFor::ForEachN(first, count, op, stream);
       cuda_cub::throw_on_error(status, "parallel_for failed");
       status = cuda_cub::synchronize_optional(policy);
       cuda_cub::throw_on_error(status, "parallel_for: failed to synchronize");),
       (for (Size idx = 0; idx != count; ++idx)
        {
          op(raw_reference_cast(*(first + idx)));
        }
    ));

    return first + count;
  }

  // for_each
  template <class Derived,
            class Input,
            class UnaryOp>
  Input THRUST_FUNCTION
  for_each(execution_policy<Derived> &policy,
           Input                      first,
           Input                      last,
           UnaryOp                    op)
  {
    typedef typename iterator_traits<Input>::difference_type size_type;
    size_type count = static_cast<size_type>(thrust::distance(first,last));

    return THRUST_NS_QUALIFIER::cuda_cub::for_each_n(policy, first, count, op);
  }
} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif
