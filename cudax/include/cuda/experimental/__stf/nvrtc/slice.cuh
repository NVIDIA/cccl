//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Core definition of `slice` that can be used from NVRTC
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/mdspan>
#include <cuda/experimental/__stf/internal/slice_core.cuh>

namespace cuda::experimental::stf
{

template <typename T, size_t M, size_t N>
class static_slice
{
public:
    using extents_t = cuda::std::extents<size_t, M, N>;
    using layout_t  = cuda::std::layout_stride;
    using mdspan_t  = cuda::std::mdspan<T, extents_t, layout_t>;

    __host__ __device__
    static_slice(T* data,
                 typename layout_t::template mapping<extents_t> mapping)      // explicit ctor
        : view_{data, mapping}
    {}

    // convert from a dynamic-extents mdspan
    template <typename OtherMapping>
    __host__ __device__
    static_slice(const cuda::std::mdspan<T,
                                         cuda::std::dextents<size_t, 2>,
                                         OtherMapping>& dyn)
        : view_{dyn.data_handle(),
                typename layout_t::template mapping<extents_t>(dyn.mapping())}
    {
        assert(dyn.extent(0) == M && dyn.extent(1) == N);
    }

    __host__ __device__       T& operator()(size_t i, size_t j)       { return view_(i, j); }
    __host__ __device__ const T& operator()(size_t i, size_t j) const { return view_(i, j); }

    __host__ __device__ T* data()      const { return view_.data_handle(); }
    __host__ __device__ auto mapping() const { return view_.mapping();     }
    __host__ __device__ auto extents() const { return view_.extents();     }
    __host__ __device__ size_t extent(size_t i) const { return view_.extent(i); }

    __host__ __device__ constexpr size_t size() const noexcept
    {
        return M * N;
    }

private:
    mdspan_t view_;
};

} // end namespace cuda::experimental::stf
