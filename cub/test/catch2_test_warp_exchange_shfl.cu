// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "catch2_test_warp_exchange.cuh"

namespace
{
using inout_types = c2h::type_list<std::uint16_t, std::int32_t, std::int64_t, double>;

using items_per_thread = c2h::enum_type_list<int, 2, 4, 8, 16, 32>;

template <class TestType>
struct params_t
{
  using in_type  = c2h::get<0, TestType>;
  using out_type = c2h::get<0, TestType>;

  static constexpr int logical_warp_threads = c2h::get<1, TestType>::value;
  static constexpr int items_per_thread     = c2h::get<1, TestType>::value;
  static constexpr int total_warps          = total_warps_t<logical_warp_threads>::value();
  static constexpr int tile_size            = logical_warp_threads * items_per_thread;
  static constexpr int total_item_count     = total_warps * tile_size;
};
} // namespace

C2H_TEST("Blocked to striped works", "[exchange][warp][shfl]", inout_types, items_per_thread)
{
  using params   = params_t<TestType>;
  using in_type  = typename params::in_type;
  using out_type = typename params::out_type;
  c2h::device_vector<out_type> d_out(params::total_item_count, out_type{});
  c2h::device_vector<in_type> d_in(params::total_item_count);

  c2h::gen(c2h::modulo_t{d_in.size()}, d_in);

  warp_exchange<params::logical_warp_threads, params::items_per_thread, params::total_warps, cub::WARP_EXCHANGE_SHUFFLE>(
    d_in, d_out, blocked_to_striped{});
  c2h::host_vector<out_type> h_expected_output(d_out.size());
  fill_striped<params::logical_warp_threads,
               params::items_per_thread,
               params::logical_warp_threads * params::total_warps>(h_expected_output.begin());

  REQUIRE(h_expected_output == d_out);
}

C2H_TEST("Striped to blocked works", "[exchange][warp][shfl]", inout_types, items_per_thread)
{
  using params   = params_t<TestType>;
  using in_type  = typename params::in_type;
  using out_type = typename params::out_type;
  c2h::device_vector<out_type> d_out(params::total_item_count, out_type{});

  c2h::host_vector<in_type> h_in(params::total_item_count);
  fill_striped<params::logical_warp_threads,
               params::items_per_thread,
               params::logical_warp_threads * params::total_warps>(h_in.begin());
  c2h::device_vector<in_type> d_in = h_in;

  warp_exchange<params::logical_warp_threads, params::items_per_thread, params::total_warps, cub::WARP_EXCHANGE_SHUFFLE>(
    d_in, d_out, striped_to_blocked{});
  c2h::device_vector<out_type> d_expected_output(d_out.size());
  c2h::gen(c2h::modulo_t{d_out.size()}, d_expected_output);

  REQUIRE(d_expected_output == d_out);
}
