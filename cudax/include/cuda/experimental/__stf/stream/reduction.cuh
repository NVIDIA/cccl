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
 * @brief Interface to define reduction operators in the CUDA stream backend
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

#include <cuda/experimental/__stf/internal/logical_data.cuh>
#include <cuda/experimental/__stf/stream/internal/event_types.cuh>

namespace cuda::experimental::stf
{
/**
 * @brief Helper class to define a reduction operator attached to a data
 * interface.
 *
 * Defining a new operator requires to define the virtual methods
 * stream_init_op and stream_redux_op which respectively initialize a data
 * instance, and apply the reduction operator over two instances.
 *
 * See stream_reduction_operator for a class which directly manipulates typed
 * data instances, with a simpler programming interface.
 */
class stream_reduction_operator_untyped : public reduction_operator_base
{
public:
  stream_reduction_operator_untyped(bool is_commutative = true)
      : reduction_operator_base(is_commutative)
  {}

  virtual void stream_redux_op(
    logical_data_untyped& d,
    const data_place& inout_memory_node,
    instance_id_t inout_instance_id,
    const data_place& in_memory_node,
    instance_id_t in_instance_id,
    const exec_place& e,
    cudaStream_t s) = 0;

  virtual void stream_init_op(
    logical_data_untyped& d,
    const data_place& out_memory_node,
    instance_id_t out_instance_id,
    const exec_place& e,
    cudaStream_t s) = 0;

#ifndef _CCCL_DOXYGEN_INVOKED // doxygen has issues with this code
  void op_untyped(
    logical_data_untyped& d,
    const data_place& inout_memory_node,
    instance_id_t inout_instance_id,
    const data_place& in_memory_node,
    instance_id_t in_instance_id,
    const exec_place& ep,
    event_list& prereqs) override
  {
    auto dstream  = inout_memory_node.getDataStream(d.get_ctx().async_resources());
    auto async_op = stream_async_op(d.get_ctx(), dstream, prereqs);
    if (d.get_ctx().generate_event_symbols())
    {
      async_op.set_symbol("redux op " + d.get_symbol());
    }

#  ifdef REDUCTION_DEBUG
    fprintf(stderr, "stream_redux_op inout %d in %d\n", inout_instance_id, in_instance_id);
#  endif
    stream_redux_op(d, inout_memory_node, inout_instance_id, in_memory_node, in_instance_id, ep, dstream.stream);

    prereqs = async_op.end(d.get_ctx());
  }

  void init_op_untyped(logical_data_untyped& d,
                       const data_place& out_memory_node,
                       instance_id_t out_instance_id,
                       const exec_place& ep,
                       event_list& prereqs) override
  {
    auto dstream  = out_memory_node.getDataStream(d.get_ctx().async_resources());
    auto async_op = stream_async_op(d.get_ctx(), dstream, prereqs);
    if (d.get_ctx().generate_event_symbols())
    {
      async_op.set_symbol("redux init op " + d.get_symbol());
    }

#  ifdef REDUCTION_DEBUG
    fprintf(stderr, "stream_init_op out %d\n", out_instance_id);
#  endif
    stream_init_op(d, out_memory_node, out_instance_id, ep, dstream.stream);

    prereqs = async_op.end(d.get_ctx());
  }
#endif // _CCCL_DOXYGEN_INVOKED // doxygen has issues with this code
};

/**
 * @class stream_reduction_operator
 *
 * @brief Helper class to define a reduction operator attached to a type of
 * data instance.
 *
 * Defining a new operator requires to define the virtual
 * methods op and init_op which respectively initialize a data
 * instance, and apply the reduction operator over two instances.
 */
template <typename T>
class stream_reduction_operator : public stream_reduction_operator_untyped
{
  /**
   * @brief Apply the reduction operator over 'in' and 'inout' data instances
   * asynchronously in cudaStream_t 's'. The operator is applied from
   * execution place 'e'
   */
  virtual void op(const T& in, T& inout, const exec_place& e, cudaStream_t s) = 0;

  /**
   * @brief Initialize data instance 'out' with a neutral element with respect to the reduction operator.
   *
   * This is done asynchronously in cudaStream_t 's'. The operator is applied
   * from execution place 'e'.
   */
  virtual void init_op(T& out, const exec_place& e, cudaStream_t s) = 0;

  void stream_redux_op(
    logical_data_untyped& d,
    const data_place& /*unused*/,
    instance_id_t inout_instance_id,
    const data_place&,
    instance_id_t in_instance_id,
    const exec_place& e,
    cudaStream_t s)
  {
    const auto& in_instance = d.instance<T>(in_instance_id);
    auto& inout_instance    = d.instance<T>(inout_instance_id);

    op(in_instance, inout_instance, e, s);
  }

  void stream_init_op(
    logical_data_untyped& d, const data_place&, instance_id_t out_instance_id, const exec_place& e, cudaStream_t s)
  {
    auto& out_instance = d.instance<T>(out_instance_id);

    init_op(out_instance, e, s);
  }
};
} // end namespace cuda::experimental::stf
