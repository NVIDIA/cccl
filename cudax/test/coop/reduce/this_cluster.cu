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

constexpr int block_size = 64;

/***********************************************************************************************************************
 * Thread Reduce Wrapper Kernels
 **********************************************************************************************************************/

template <bool Broadcasted>
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
    cudax::this_cluster cluster{config};

    T thread_data[NumItems];
    for (int i = 0; i < NumItems; ++i)
    {
      thread_data[i] = d_in[cuda::gpu_thread.rank_as<int>(cluster) + i * cuda::gpu_thread.count_as<int>(cluster)];
    }

    if constexpr (Broadcasted)
    {
      const auto result = cudax::coop::reduce(cudax::broadcasted, cluster, thread_data, red_op);

      d_out[cuda::gpu_thread.rank(cluster)] = result;
    }
    else
    {
      const auto result = cudax::coop::reduce(cluster, thread_data, red_op);

      REQUIRE(result.has_value() == cuda::gpu_thread.is_root_rank(cluster));
      if (cuda::gpu_thread.is_root_rank(cluster))
      {
        *d_out = result.value();
      }
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

using cluster_size_list = c2h::enum_type_list<int, 1, 2, 4, 8>;

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

template <int ClusterSize, class T, class RedOp, bool Broadcasted = false>
void run_reduce_kernel(
  cuda::stream_ref stream,
  cuda::std::integral_constant<int, ClusterSize>,
  int num_items,
  const c2h::device_vector<T>& in,
  c2h::device_vector<T>& out,
  RedOp red_op,
  cuda::std::bool_constant<Broadcasted> = {})
{
  const auto config =
    cuda::make_config(cuda::grid_dims<1>(), cuda::cluster_dims<ClusterSize>(), cuda::block_dims<block_size>());
  const auto in_ptr  = thrust::raw_pointer_cast(in.data());
  const auto out_ptr = thrust::raw_pointer_cast(out.data());
  const ReduceKernel<Broadcasted> kernel{};

  switch (num_items)
  {
    case 1:
      cuda::launch(stream, config, kernel, cuda::std::integral_constant<int, 1>{}, in_ptr, out_ptr, red_op);
      break;
    case 4:
      cuda::launch(stream, config, kernel, cuda::std::integral_constant<int, 4>{}, in_ptr, out_ptr, red_op);
      break;
    default:
      FAIL("Unsupported number of items");
  }
  stream.sync();
}

constexpr int max_size  = 4;
constexpr int num_seeds = 10;

/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/

_CCCL_DIAG_SUPPRESS_MSVC(4244) // warning C4244: '=': conversion from 'int' to '_Tp', possible loss of data

C2H_TEST("reduce/this_cluster Integral Type Tests",
         "[reduce][this_cluster]",
         integral_type_list,
         operator_integral_list,
         cluster_size_list)
{
  const auto device = cuda::devices[0];
  if (cuda::device_attributes::compute_capability_major(device) < 9)
  {
    return;
  }

  using value_t                    = c2h::get<0, TestType>;
  using op_t                       = c2h::get<1, TestType>;
  using cluster_size_t             = c2h::get<2, TestType>;
  constexpr auto reduce_op         = op_t{};
  constexpr auto operator_identity = cuda::identity_element<op_t, value_t>();
  CAPTURE(c2h::type_name<value_t>(), max_size, c2h::type_name<decltype(reduce_op)>());
  c2h::device_vector<value_t> d_in(max_size * cluster_size_t::value * block_size);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, cuda::std::numeric_limits<value_t>::min());
  c2h::host_vector<value_t> h_in = d_in;
  cuda::stream stream{device};
  for (int num_items : {1, 4})
  {
    auto reference_result = cuda::std::accumulate(
      h_in.begin(), h_in.begin() + num_items * cluster_size_t::value * block_size, operator_identity, reduce_op);
    run_reduce_kernel(stream, cluster_size_t{}, num_items, d_in, d_out, reduce_op);
    verify_results(reference_result, c2h::host_vector<value_t>(d_out)[0]);
  }
}

C2H_TEST("reduce/this_cluster Floating-Point Type Tests",
         "[reduce][this_cluster]",
         fp_type_list,
         operator_fp_list,
         cluster_size_list)
{
  const auto device = cuda::devices[0];
  if (cuda::device_attributes::compute_capability_major(device) < 9)
  {
    return;
  }

  using value_t                = c2h::get<0, TestType>;
  using op_t                   = c2h::get<1, TestType>;
  using cluster_size_t         = c2h::get<2, TestType>;
  constexpr auto reduce_op     = op_t{};
  const auto operator_identity = cuda::identity_element<op_t, value_t>();
  CAPTURE(c2h::type_name<value_t>(), max_size, c2h::type_name<decltype(reduce_op)>());
  c2h::device_vector<value_t> d_in(max_size * cluster_size_t::value * block_size);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, cuda::std::numeric_limits<value_t>::min());
  c2h::host_vector<value_t> h_in = d_in;
  cuda::stream stream{device};
  for (int num_items : {1, 4})
  {
    auto reference_result = cuda::std::accumulate(
      h_in.begin(), h_in.begin() + num_items * cluster_size_t::value * block_size, operator_identity, reduce_op);
    run_reduce_kernel(stream, cluster_size_t{}, num_items, d_in, d_out, reduce_op);
    verify_results(reference_result, c2h::host_vector<value_t>(d_out)[0]);
  }
}

C2H_TEST("reduce/this_cluster Broadcasted", "[reduce][this_cluster]", integral_type_list, cluster_size_list)
{
  const auto device = cuda::devices[0];
  if (cuda::device_attributes::compute_capability_major(device) < 9)
  {
    return;
  }

  using value_t                    = c2h::get<0, TestType>;
  using op_t                       = cuda::std::plus<>;
  using cluster_size_t             = c2h::get<1, TestType>;
  constexpr auto reduce_op         = op_t{};
  constexpr auto operator_identity = cuda::identity_element<op_t, value_t>();
  CAPTURE(c2h::type_name<value_t>(), max_size, c2h::type_name<decltype(reduce_op)>());
  c2h::device_vector<value_t> d_in(max_size * cluster_size_t::value * block_size);
  c2h::gen(C2H_SEED(num_seeds), d_in, cuda::std::numeric_limits<value_t>::min());
  c2h::host_vector<value_t> h_in = d_in;
  cuda::stream stream{device};
  for (int num_items : {1, 4})
  {
    c2h::device_vector<value_t> d_out(cluster_size_t::value * block_size);
    auto reference_result = cuda::std::accumulate(
      h_in.begin(), h_in.begin() + num_items * cluster_size_t::value * block_size, operator_identity, reduce_op);
    run_reduce_kernel(stream, cluster_size_t{}, num_items, d_in, d_out, reduce_op, cuda::std::true_type{});
    verify_results(c2h::host_vector<value_t>(cluster_size_t::value * block_size, reference_result),
                   c2h::host_vector<value_t>(d_out));
  }
}
