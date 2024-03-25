
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
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/transform.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

namespace __copy {
  template <class Derived,
            class InputIt,
            class OutputIt>
  OutputIt THRUST_RUNTIME_FUNCTION
  device_to_device(execution_policy<Derived>& policy,
                   InputIt                    first,
                   InputIt                    last,
                   OutputIt                   result,
                   thrust::detail::true_type)
  {
    typedef typename thrust::iterator_traits<InputIt>::value_type InputTy;
    const auto n = thrust::distance(first, last);
    if (n > 0) {
      cudaError status;
      status = trivial_copy_device_to_device(policy,
                                             reinterpret_cast<InputTy*>(thrust::raw_pointer_cast(&*result)),
                                             reinterpret_cast<InputTy const*>(thrust::raw_pointer_cast(&*first)),
                                             n);
      cuda_cub::throw_on_error(status, "__copy:: D->D: failed");
    }

    return result + n;
  }

  template <class Derived,
            class InputIt,
            class OutputIt>
  OutputIt THRUST_RUNTIME_FUNCTION
  device_to_device(execution_policy<Derived>& policy,
                   InputIt                    first,
                   InputIt                    last,
                   OutputIt                   result,
                   thrust::detail::false_type)
  {
    typedef typename thrust::iterator_traits<InputIt>::value_type InputTy;
    return cuda_cub::transform(policy,
                              first,
                              last,
                              result,
                              thrust::identity<InputTy>());
  }

  template <class Derived,
            class InputIt,
            class OutputIt>
  OutputIt THRUST_RUNTIME_FUNCTION
  device_to_device(execution_policy<Derived>& policy,
                   InputIt                    first,
                   InputIt                    last,
                   OutputIt                   result)
  {
    return device_to_device(policy,
                            first,
                            last,
                            result,
                            typename is_indirectly_trivially_relocatable_to<InputIt, OutputIt>::type());
  }
}    // namespace __copy

}    // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
