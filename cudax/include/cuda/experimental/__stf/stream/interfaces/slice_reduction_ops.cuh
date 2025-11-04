//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Reduction operators over slices in the CUDA stream backend
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

#include <cuda/experimental/__stf/stream/reduction.cuh>

namespace cuda::experimental::stf
{
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
template <typename element_type, size_t dimensions = 1, typename ReduxOp>
__global__ void
slice_reduction_op_kernel(const slice<element_type, dimensions> in, const slice<element_type, dimensions> inout)
{
  size_t tid      = threadIdx.x + blockIdx.x * blockDim.x;
  size_t nthreads = blockDim.x * gridDim.x;

  if constexpr (dimensions == 1)
  {
    for (size_t i = tid; i < inout.extent(0); i += nthreads)
    {
      ReduxOp::op_gpu(in(i), inout(i));
    }
  }
  else if constexpr (dimensions == 2)
  {
    for (size_t j = 0; j < inout.extent(1); j++)
    {
      for (size_t i = tid; i < inout.extent(0); i += nthreads)
      {
        ReduxOp::op_gpu(in(i, j), inout(i, j));
      }
    }
  }
  else
  {
    static_assert(dimensions == 1 || dimensions == 2, "Dimensionality not supported.");
  }
}

template <typename element_type, size_t dimensions = 1, typename ReduxOp>
__global__ void slice_reduction_op_init_kernel(slice<element_type, dimensions> out)
{
  size_t tid      = threadIdx.x + blockIdx.x * blockDim.x;
  size_t nthreads = blockDim.x * gridDim.x;

  if constexpr (dimensions == 1)
  {
    for (size_t i = tid; i < out.extent(0); i += nthreads)
    {
      ReduxOp::init_gpu(out(i));
    }
  }
  else if constexpr (dimensions == 2)
  {
    for (size_t j = 0; j < out.extent(1); j++)
    {
      for (size_t i = tid; i < out.extent(0); i += nthreads)
      {
        ReduxOp::init_gpu(out(i, j));
      }
    }
  }
  else
  {
    static_assert(dimensions == 1 || dimensions == 2, "Dimensionality not supported.");
  }
}
#endif // !_CCCL_DOXYGEN_INVOKED

/**
 * @brief Helper class to define element-wise reduction operators applied to slices
 *
 *   ReduxOp::init_host(element_type &out);
 *   ReduxOp::op_host(const element_type &in, element_type &inout);
 *   __device__ ReduxOp::init_gpu(element_type &out);
 *   __device__ ReduxOp::op_gpu(const element_type &in, element_type &inout);
 *
 * @extends stream_reduction_operator
 */
template <typename element_type, size_t dimensions, typename ReduxOp>
class slice_reduction_op : public stream_reduction_operator<slice<element_type, dimensions>>
{
public:
  using instance_t = slice<element_type, dimensions>;

  slice_reduction_op() = default;

  /// Reconstruct an instance by applying the reduction operator over it and another instance
  void op(const instance_t& in, instance_t& inout, const exec_place& e, cudaStream_t s) override
  {
    if (e.affine_data_place().is_host())
    {
      // TODO make a callback when the situation gets better
      cuda_safe_call(cudaStreamSynchronize(s));
      // slice_print(in, "in before op");
      // slice_print(inout, "inout before op");

      if constexpr (dimensions == 1)
      {
        assert(in.extent(0) == inout.extent(0));
        for (size_t i = 0; i < in.extent(0); i++)
        {
          ReduxOp::op_host(in(i), inout(i));
        }
      }
      else if constexpr (dimensions == 2)
      {
        for (size_t j = 0; j < inout.extent(1); j++)
        {
          for (size_t i = 0; i < inout.extent(0); i++)
          {
            ReduxOp::op_host(in(i, j), inout(i, j));
          }
        }
      }
      else
      {
        static_assert(dimensions == 1 || dimensions == 2, "Dimensionality not supported.");
      }

      // slice_print(in, "in after op");
      // slice_print(inout, "inout after op");
    }
    else
    {
      // this is not the host, so this has to be a device ... (XXX)
      const auto occ = reserved::compute_occupancy(slice_reduction_op_kernel<element_type, dimensions, ReduxOp>);
      slice_reduction_op_kernel<element_type, dimensions, ReduxOp>
        <<<occ.min_grid_size, occ.block_size, 0, s>>>(in, inout);
    }
  }

  /// Initialize an instance with an appropriate default value for the reduction operator
  void init_op(instance_t& out, const exec_place& e, cudaStream_t s) override
  {
    if (e.affine_data_place().is_host())
    {
      // TODO make a callback when the situation gets better
      cuda_safe_call(cudaStreamSynchronize(s));
      if constexpr (dimensions == 1)
      {
        for (size_t i = 0; i < out.extent(0); i++)
        {
          ReduxOp::init_host(out(i));
        }
      }
      else if constexpr (dimensions == 2)
      {
        for (size_t j = 0; j < out.extent(1); j++)
        {
          for (size_t i = 0; i < out.extent(0); i++)
          {
            ReduxOp::init_host(out(i, j));
          }
        }
      }
      else
      {
        static_assert(dimensions == 1 || dimensions == 2, "Dimensionality not supported.");
      }
    }
    else
    {
      // this is not the host, so this has to be a device ... (XXX)
      const auto occ = reserved::compute_occupancy(slice_reduction_op_init_kernel<element_type, dimensions, ReduxOp>);

      EXPECT(out.data_handle() != nullptr);
      slice_reduction_op_init_kernel<element_type, dimensions, ReduxOp>
        <<<occ.min_grid_size, occ.block_size, 0, s>>>(out);
    }
  }
};

template <typename element_type>
class slice_reduction_op_sum_impl
{
public:
  static void init_host(element_type& out)
  {
    out = element_type(0);
  };
  static __device__ void init_gpu(element_type& out)
  {
    out = element_type(0);
  };

  static void op_host(const element_type& in, element_type& inout)
  {
    inout += in;
  };
  static __device__ void op_gpu(const element_type& in, element_type& inout)
  {
    inout += in;
  };
};

/**
 * @brief A sum reduction operator over slices
 */

template <typename element_type, size_t dimensions = 1>
class slice_reduction_op_sum
    : public slice_reduction_op<element_type, dimensions, slice_reduction_op_sum_impl<element_type>>
{};
} // end namespace cuda::experimental::stf
