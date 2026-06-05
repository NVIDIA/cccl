//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/functional>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/coop.cuh>
#include <cuda/experimental/group.cuh>

#include <testing.cuh>

#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>
#include <c2h/generators.h>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

/***********************************************************************************************************************
 * Thread Reduce Wrapper Kernels
 **********************************************************************************************************************/

struct ReduceKernel
{
  template <class Config, int NumItems, class T, class RedOp>
  __device__ void operator()(
    Config config,
    cuda::std::integral_constant<int, NumItems>,
    const T* __restrict__ d_in,
    T* __restrict__ d_out,
    RedOp red_op)
  {
    cudax::this_warp warp{config};

    T thread_data[NumItems];
    for (int i = 0; i < NumItems; ++i)
    {
      thread_data[i] = d_in[cuda::gpu_thread.rank_as<int>(warp) + i * cuda::gpu_thread.count_as<int>(warp)];
    }
    const auto result = cudax::coop::reduce(warp, thread_data, red_op);

    REQUIRE(result.has_value() == cuda::gpu_thread.is_root_rank(warp));
    if (cuda::gpu_thread.is_root_rank(warp))
    {
      *d_out = result.value();
    }
  }
};

/***********************************************************************************************************************
 * Type list definition
 **********************************************************************************************************************/

using integral_type_list =
  c2h::type_list<cuda::std::int8_t, cuda::std::int16_t, cuda::std::uint16_t, cuda::std::int32_t, cuda::std::int64_t>;

using fp_type_list = c2h::type_list<float, double>;

using operator_integral_list =
  c2h::type_list<cuda::std::plus<>,
                 cuda::std::multiplies<>,
                 cuda::std::bit_and<>,
                 cuda::std::bit_or<>,
                 cuda::std::bit_xor<>,
                 cuda::minimum<>,
                 cuda::maximum<>>;

using operator_fp_list = c2h::type_list<cuda::std::plus<>, cuda::std::multiplies<>, cuda::minimum<>, cuda::maximum<>>;

/***********************************************************************************************************************
 * Verify results and kernel launch
 **********************************************************************************************************************/

template <class T>
void verify_results(const T& expected_data, const T& test_results)
{
  if constexpr (cuda::std::is_floating_point_v<T>)
  {
    REQUIRE_THAT(expected_data, Catch::Matchers::WithinRel(test_results, T{0.05}));
  }
  else
  {
    REQUIRE(expected_data == test_results);
  }
}

template <class T, class RedOp>
void run_thread_reduce_kernel(
  cuda::stream_ref stream, int num_items, const c2h::device_vector<T>& in, c2h::device_vector<T>& out, RedOp red_op)
{
  const auto config  = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<32>());
  const auto in_ptr  = thrust::raw_pointer_cast(in.data());
  const auto out_ptr = thrust::raw_pointer_cast(out.data());
  const ReduceKernel kernel{};

  switch (num_items)
  {
    case 1:
      cuda::launch(stream, config, kernel, cuda::std::integral_constant<int, 1>{}, in_ptr, out_ptr, red_op);
      break;
    case 2:
      cuda::launch(stream, config, kernel, cuda::std::integral_constant<int, 2>{}, in_ptr, out_ptr, red_op);
      break;
    case 3:
      cuda::launch(stream, config, kernel, cuda::std::integral_constant<int, 3>{}, in_ptr, out_ptr, red_op);
      break;
    case 4:
      cuda::launch(stream, config, kernel, cuda::std::integral_constant<int, 4>{}, in_ptr, out_ptr, red_op);
      break;
    default:
      FAIL("Unsupported number of items");
  }
  stream.sync();
}

constexpr int warp_size = 32;
constexpr int max_size  = 4;
constexpr int num_seeds = 10;

/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/

_CCCL_DIAG_SUPPRESS_MSVC(4244) // warning C4244: '=': conversion from 'int' to '_Tp', possible loss of data

C2H_TEST("reduce/this_warp Integral Type Tests", "[reduce][this_warp]", integral_type_list, operator_integral_list)
{
  using value_t                    = c2h::get<0, TestType>;
  using op_t                       = c2h::get<1, TestType>;
  constexpr auto reduce_op         = op_t{};
  constexpr auto operator_identity = cuda::identity_element<op_t, value_t>();
  CAPTURE(c2h::type_name<value_t>(), max_size, c2h::type_name<decltype(reduce_op)>());
  c2h::device_vector<value_t> d_in(max_size * warp_size);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, cuda::std::numeric_limits<value_t>::min());
  c2h::host_vector<value_t> h_in = d_in;
  cuda::stream stream{cuda::devices[0]};
  for (int num_items = 1; num_items <= max_size; ++num_items)
  {
    auto reference_result =
      cuda::std::accumulate(h_in.begin(), h_in.begin() + num_items * warp_size, operator_identity, reduce_op);
    run_thread_reduce_kernel(stream, num_items, d_in, d_out, reduce_op);
    verify_results(reference_result, c2h::host_vector<value_t>(d_out)[0]);
  }
}

C2H_TEST("reduce/this_warp Floating-Point Type Tests", "[reduce][this_warp]", fp_type_list, operator_fp_list)
{
  using value_t                = c2h::get<0, TestType>;
  using op_t                   = c2h::get<1, TestType>;
  constexpr auto reduce_op     = op_t{};
  const auto operator_identity = cuda::identity_element<op_t, value_t>();
  CAPTURE(c2h::type_name<value_t>(), max_size, c2h::type_name<decltype(reduce_op)>());
  c2h::device_vector<value_t> d_in(max_size * warp_size);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, cuda::std::numeric_limits<value_t>::min());
  c2h::host_vector<value_t> h_in = d_in;
  cuda::stream stream{cuda::devices[0]};
  for (int num_items = 1; num_items <= max_size; ++num_items)
  {
    auto reference_result =
      cuda::std::accumulate(h_in.begin(), h_in.begin() + num_items * warp_size, operator_identity, reduce_op);
    run_thread_reduce_kernel(stream, num_items, d_in, d_out, reduce_op);
    verify_results(reference_result, c2h::host_vector<value_t>(d_out)[0]);
  }
}
