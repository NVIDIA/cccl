//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/stream/reduction.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <iostream>

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
  *inout_addr += *in_addr;
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
  stream_ctx ctx;

  const int N = 128;

  // We have an array, and a handle for each entry of the array
  int array[N];
  logical_data<slice<int>> array_handles[N];

  /*
   * We are going to compute the sum of this array
   */
  for (int i = 0; i < N; i++)
  {
    array[i]         = i;
    array_handles[i] = ctx.logical_data(&array[i], {1});
    array_handles[i].set_symbol(std::string("array[") + std::to_string(i) + std::string("]"));
  }

  logical_data<slice<int>> var_handle = ctx.logical_data(shape_of<slice<int>>(1));
  var_handle.set_symbol("var");

  int check_sum = 0;
  for (int i = 0; i < N; i++)
  {
    check_sum += array[i];
  }

  auto redux_op = std::make_shared<scalar_sum_t>();

  for (int i = 0; i < N; i++)
  {
    ctx.task(var_handle.relaxed(redux_op), array_handles[i].read())
        ->*[](cudaStream_t stream, auto d_var, auto d_array_i) {
              add<<<1, 1, 0, stream>>>(d_array_i.data_handle(), d_var.data_handle());
            };
  }

  // Force the reconstruction of data on the device, so that no transfers are
  // necessary while reconstructing the result.
  // This will of course not be necessary in the future ...
  ctx.task(var_handle.read())->*[](cudaStream_t /*unused*/, auto /*unused*/) {};

  // Check result
  ctx.task(exec_place::host, var_handle.read())->*[=](cudaStream_t stream, auto h_var) {
    cuda_safe_call(cudaStreamSynchronize(stream));
    int value = h_var(0);
    EXPECT(value == check_sum);
  };

  ctx.finalize();
}
