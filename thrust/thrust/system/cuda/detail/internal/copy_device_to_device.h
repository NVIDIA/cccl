
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

#if _CCCL_HAS_CUDA_COMPILER()
#  include <thrust/system/cuda/config.h>

#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/transform.h>
#  include <thrust/system/cuda/detail/util.h>
#  include <thrust/type_traits/is_trivially_relocatable.h>

#  include <cuda/std/iterator>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
// Need a forward declaration here to work around a cyclic include, since "cuda/detail/transform.h" includes this header
template <class Derived, class InputIt, class OutputIt, class TransformOp>
OutputIt THRUST_FUNCTION
transform(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, TransformOp transform_op);

namespace __copy
{
template <class Derived, class InputIt, class OutputIt>
OutputIt THRUST_RUNTIME_FUNCTION
device_to_device(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result)
{
  // FIXME(bgruber): We must not check is_trivially_relocatable, since we do not semantically relocate, but copy (the
  // source remains valid). This is relevant for types like `unique_ptr`, which are trivially relocatable, but not
  // trivially copyable.
  if constexpr (is_indirectly_trivially_relocatable_to<InputIt, OutputIt>::value)
  {
    using InputTy = thrust::detail::it_value_t<InputIt>;
    const auto n  = ::cuda::std::distance(first, last);
    if (n > 0)
    {
      const cudaError status = trivial_copy_device_to_device(
        policy,
        reinterpret_cast<InputTy*>(thrust::raw_pointer_cast(&*result)),
        reinterpret_cast<InputTy const*>(thrust::raw_pointer_cast(&*first)),
        n);
      cuda_cub::throw_on_error(status, "__copy:: D->D: failed");
    }

    return result + n;
  }
  else
  {
    return cuda_cub::transform(policy, first, last, result, ::cuda::std::identity{});
  }
}
} // namespace __copy

} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
