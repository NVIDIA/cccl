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
 * @brief Use the reduction access mode to add variables concurrently on different places
 */

#include <cuda/experimental/__stf/stream/reduction.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

using scalar_t = slice_stream_interface<int, 1>;

template <typename T>
__global__ void set_value(T* addr, T val)
{
  *addr = val;
}

template <typename T>
__global__ void add(const T* in_addr, T* inout_addr)
{
  // printf("ADD: %d += %d : RES %d\n", *inout_addr, *in_addr, *inout_addr + *in_addr);

  *inout_addr += *in_addr;
}

template <typename T>
__global__ void add_val(T* inout_addr, T val)
{
  *inout_addr += val;
}

/*
 * Define a SUM reduction operator over a scalar
 */
class scalar_sum_t : public stream_reduction_operator_untyped
{
public:
  scalar_sum_t()
      : stream_reduction_operator_untyped() {};

  void stream_redux_op(
    logical_data_untyped& d,
    const data_place& /*unused*/,
    instance_id_t inout_instance_id,
    const data_place& /*unused*/,
    instance_id_t in_instance_id,
    const exec_place& /*unused*/,
    cudaStream_t s) override
  {
    auto& in_instance    = d.instance<typename scalar_t::element_type>(in_instance_id);
    auto& inout_instance = d.instance<typename scalar_t::element_type>(inout_instance_id);
    // fprintf(stderr, "REDUX OP d %p inout (node %d id %d addr %p) in (node %d id %d addr %p)\n", d,
    //        inout_memory_node, inout_instance_id, *inout_instance, in_memory_node, in_instance_id, *in_instance);
    add<<<1, 1, 0, s>>>(in_instance.data_handle(), inout_instance.data_handle());
  }

  void stream_init_op(logical_data_untyped& d,
                      const data_place& /*unused*/,
                      instance_id_t out_instance_id,
                      const exec_place& /*unused*/,
                      cudaStream_t s) override
  {
    auto& out_instance = d.instance<typename scalar_t::element_type>(out_instance_id);
    // fprintf(stderr, "REDUX INIT d %p memory node %d instance id %d => addr %p\n", d, out_memory_node,
    //        out_instance_id, *out_instance);
    set_value<<<1, 1, 0, s>>>(out_instance.data_handle(), 0);
  }
};

int main()
{
  const int N = 4;

  stream_ctx ctx;

  auto var_handle = ctx.logical_data(shape_of<slice<int>>(1));
  var_handle.set_symbol("var");

  auto redux_op = std::make_shared<scalar_sum_t>();

  // We add i (total = N(N-1)/2 + initial_value)
  for (int i = 0; i < N; i++)
  {
    ctx.task(var_handle.relaxed(redux_op))->*[&](cudaStream_t stream, auto d_var) {
      add_val<<<1, 1, 0, stream>>>(d_var.data_handle(), i);
    };
  }

  // Check result
  ctx.task(exec_place::host, var_handle.read())->*[&](cudaStream_t stream, auto h_var) {
    cuda_safe_call(cudaStreamSynchronize(stream));
    int expected = (N * (N - 1)) / 2;
    EXPECT(h_var(0) == expected);
  };

  ctx.finalize();
}
