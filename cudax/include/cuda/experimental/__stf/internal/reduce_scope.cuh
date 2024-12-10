//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/execution_space.h>

#include <cuda/experimental/__stf/internal/parallel_for_scope.cuh> // for null_partition

namespace cuda::experimental::stf
{

template <typename T>
struct scalar_view;

#if !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

namespace reserved
{

/**
 * @brief Supporting class for the reduce construct
 *
 * This is used to implement operators such as ->* on the object produced by `ctx.reduce`
 *
 * @tparam deps_t
 */
template <typename context, typename shape_t, typename partitioner_t, typename reduce_op_t, typename... deps_ops_t>
class reduce_scope
{
public:
  /// @brief Constructor
  /// @param ctx Reference to context (it will not be copied, so careful with lifetimes)
  /// @param e_place Execution place for this parallel_for
  /// @param shape Shape to iterate
  /// @param ...deps Dependencies
  reduce_scope(context& ctx, exec_place e_place, shape_t shape, reduce_op_t, deps_ops_t... deps)
      : result(ctx.logical_data(shape_of<scalar_view<typename reduce_op_t::scalar_t>>()))
      , pfor_scope(ctx, e_place, shape, deps..., result.reduce(reduce_op_t{}))
  {}

  reduce_scope(const reduce_scope&)            = delete;
  reduce_scope(reduce_scope&&)                 = default;
  reduce_scope& operator=(const reduce_scope&) = delete;

  /**
   * @brief Retrieves the symbol associated with the task.
   *
   * @return A constant reference to the symbol string.
   */
  const ::std::string& get_symbol() const
  {
    return symbol;
  }

  /**
   * @brief Sets the symbol associated with the task.
   *
   * This method uses a custom move function `mv` to handle the transfer of ownership.
   *
   * @param s The new symbol string.
   * @return A reference to the current object, allowing for method chaining.
   */
  auto& set_symbol(::std::string s)
  {
    symbol = mv(s);
    return *this;
  }

  /**
   * @brief Overloads the `operator->*` to perform parallel computations using a user-defined function or lambda.
   *
   * Lower down to the corresponding parallel_for construct
   */
  template <typename Fun>
  auto operator->*(Fun&& f)
  {
      pfor_scope->*f;
      return result;
  }

private:
  ::std::string symbol;
  // ctx.reduce is lowered down to a parallel for internally
  logical_data<scalar_view<typename reduce_op_t::scalar_t>> result;
  parallel_for_scope<context, shape_t, partitioner_t, deps_ops_t...> pfor_scope;
};
} // end namespace reserved

#endif // !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)

} // end namespace cuda::experimental::stf
