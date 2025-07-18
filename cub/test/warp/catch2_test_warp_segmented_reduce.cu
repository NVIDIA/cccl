// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <cub/util_macro.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <cuda/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cstdint>

#include <c2h/catch2_test_helper.h>
#include <c2h/check_results.cuh>
#include <c2h/custom_type.h>

template <int LOGICAL_WARP_THREADS, int TOTAL_WARPS, typename T, typename ActionT>
__global__ void warp_reduce_kernel(T* in, T* out, ActionT action)
{
  using warp_reduce_t = cub::WarpReduce<T, LOGICAL_WARP_THREADS>;
  using storage_t     = typename warp_reduce_t::TempStorage;
  __shared__ storage_t storage[TOTAL_WARPS];

  auto warp_id     = threadIdx.x / LOGICAL_WARP_THREADS;
  auto thread_data = in[threadIdx.x];
  // Instantiate and run warp reduction
  if constexpr (LOGICAL_WARP_THREADS * TOTAL_WARPS % 32 == 0)
  {
    _CCCL_ASSERT(__activemask() == 0xFFFFFFFF, "invalid warp id");
  }
  warp_reduce_t warp_reduce(storage[warp_id]);
  out[threadIdx.x] = action(threadIdx.x, warp_reduce, thread_data);
}

template <typename T>
struct warp_seg_sum_tail_t
{
  uint8_t* d_flags;
  template <int LOGICAL_WARP_THREADS>
  __device__ T operator()(int linear_tid, cub::WarpReduce<T, LOGICAL_WARP_THREADS>& warp_reduce, T& thread_data) const
  {
    const bool has_agg = (linear_tid % LOGICAL_WARP_THREADS == 0) || ((linear_tid == 0) ? 0 : d_flags[linear_tid - 1]);
    auto result        = warp_reduce.TailSegmentedSum(thread_data, d_flags[linear_tid]);
    return has_agg ? result : thread_data;
  }
};

template <typename T>
struct warp_seg_sum_head_t
{
  uint8_t* d_flags;
  template <int LOGICAL_WARP_THREADS>
  __device__ T operator()(int linear_tid, cub::WarpReduce<T, LOGICAL_WARP_THREADS>& warp_reduce, T& thread_data) const
  {
    const bool has_agg = ((linear_tid % LOGICAL_WARP_THREADS == 0) || d_flags[linear_tid]);
    auto result        = warp_reduce.HeadSegmentedSum(thread_data, d_flags[linear_tid]);
    return (has_agg) ? result : thread_data;
  }
};

template <typename T, typename ReductionOpT>
struct warp_seg_reduce_tail_t
{
  uint8_t* d_flags;
  ReductionOpT reduction_op;
  template <int LOGICAL_WARP_THREADS>
  __device__ T operator()(int linear_tid, cub::WarpReduce<T, LOGICAL_WARP_THREADS>& warp_reduce, T& thread_data) const
  {
    const bool has_agg = (linear_tid % LOGICAL_WARP_THREADS == 0) || ((linear_tid == 0) ? 0 : d_flags[linear_tid - 1]);
    auto result        = warp_reduce.TailSegmentedReduce(thread_data, d_flags[linear_tid], reduction_op);
    return has_agg ? result : thread_data;
  }
};

template <typename T, typename ReductionOpT>
struct warp_seg_reduce_head_t
{
  uint8_t* d_flags;
  ReductionOpT reduction_op;
  template <int LOGICAL_WARP_THREADS>
  __device__ T operator()(int linear_tid, cub::WarpReduce<T, LOGICAL_WARP_THREADS>& warp_reduce, T& thread_data) const
  {
    const bool has_agg = ((linear_tid % LOGICAL_WARP_THREADS == 0) || d_flags[linear_tid]);
    auto result        = warp_reduce.HeadSegmentedReduce(thread_data, d_flags[linear_tid], reduction_op);
    return (has_agg) ? result : thread_data;
  }
};

/**
 * @brief Dispatch helper function
 */
template <int LOGICAL_WARP_THREADS, int TOTAL_WARPS, typename T, typename ActionT>
void warp_reduce(c2h::device_vector<T>& in, c2h::device_vector<T>& out, ActionT action)
{
  warp_reduce_kernel<LOGICAL_WARP_THREADS, TOTAL_WARPS, T, ActionT><<<1, LOGICAL_WARP_THREADS * TOTAL_WARPS>>>(
    thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), action);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

/***********************************************************************************************************************
 * Types
 **********************************************************************************************************************/

enum class reduce_mode
{
  all,
  partial,
  head_flags,
  tail_flags,
};

using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t, c2h::lexicographical_less_comparable_t>;

using full_type_list =
  c2h::type_list<uint8_t,
                 uint16_t,
                 int32_t,
                 int64_t,
                 float,
                 custom_t
#if _CCCL_CTK_AT_LEAST(13, 0)
                 ,
                 ulonglong4_16a
#else // _CCCL_CTK_AT_LEAST(13, 0)
                 ,
                 ulonglong4
#endif // _CCCL_CTK_AT_LEAST(13, 0)>;
                 >;

using builtin_type_list = c2h::type_list<int8_t, uint16_t, int32_t, int64_t, float, double>;

using logical_warp_threads = c2h::enum_type_list<int, 32, 16, 7, 1>;

using segmented_modes = c2h::enum_type_list<reduce_mode, reduce_mode::head_flags, reduce_mode::tail_flags>;

using op_list = c2h::type_list<cuda::std::plus<>, cuda::minimum<>>;

using flag_t = uint8_t;

template <int logical_warp_threads>
struct total_warps_t
{
private:
  static constexpr int max_warps      = 2;
  static constexpr bool is_arch_warp  = (logical_warp_threads == cub::detail::warp_threads);
  static constexpr bool is_pow_of_two = ((logical_warp_threads & (logical_warp_threads - 1)) == 0);
  static constexpr int total_warps    = (is_arch_warp || is_pow_of_two) ? max_warps : 1;

public:
  static constexpr int value()
  {
    return total_warps;
  }
};

template <typename TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int logical_warp_threads = c2h::get<1, TestType>::value;
  static constexpr int total_warps          = total_warps_t<logical_warp_threads>::value();
  static constexpr int tile_size            = total_warps * logical_warp_threads;
};

/***********************************************************************************************************************
 * Reference
 **********************************************************************************************************************/

template <typename InputItT, typename FlagInputItT, typename ReductionOp, typename ResultOutItT>
void compute_host_reference(
  reduce_mode mode,
  InputItT h_in,
  FlagInputItT h_flags,
  int warps,
  int logical_warp_threads,
  int valid_warp_threads,
  ReductionOp reduction_op,
  ResultOutItT h_data_out)
{
  // Accumulate segments (lane 0 of each warp is implicitly a segment head)
  for (int warp = 0; warp < warps; ++warp)
  {
    int warp_offset = warp * logical_warp_threads;
    int item_offset = warp_offset + valid_warp_threads - 1;
    // Last item in warp
    auto head_aggregate = h_in[item_offset];
    auto tail_aggregate = h_in[item_offset];
    if (mode != reduce_mode::tail_flags && h_flags[item_offset])
    {
      h_data_out[item_offset] = head_aggregate;
    }
    item_offset--;
    // Work backwards
    while (item_offset >= warp_offset)
    {
      if (h_flags[item_offset + 1])
      {
        head_aggregate = h_in[item_offset];
      }
      else
      {
        head_aggregate = reduction_op(head_aggregate, h_in[item_offset]);
      }

      if (h_flags[item_offset])
      {
        if (mode == reduce_mode::head_flags)
        {
          h_data_out[item_offset] = head_aggregate;
        }
        else if (mode == reduce_mode::tail_flags)
        {
          h_data_out[item_offset + 1] = tail_aggregate;
          tail_aggregate              = h_in[item_offset];
        }
      }
      else
      {
        tail_aggregate = reduction_op(tail_aggregate, h_in[item_offset]);
      }

      item_offset--;
    }
    // Record last segment aggregate
    if (mode == reduce_mode::tail_flags)
    {
      h_data_out[warp_offset] = tail_aggregate;
    }
    else
    {
      h_data_out[warp_offset] = head_aggregate;
    }
  }
}

/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/

C2H_TEST(
  "WarpReduce::Segmented::Sum", "[reduce][warp][segmented]", full_type_list, logical_warp_threads, segmented_modes)
{
  using params                 = params_t<TestType>;
  using T                      = typename params::type;
  constexpr auto segmented_mod = c2h::get<2, TestType>::value;
  CAPTURE(c2h::type_name<T>(), params::logical_warp_threads, segmented_mod, params::total_warps);
  using warp_seg_sum_t =
    cuda::std::_If<(segmented_mod == reduce_mode::tail_flags), warp_seg_sum_tail_t<T>, warp_seg_sum_head_t<T>>;

  c2h::device_vector<T> d_in(params::tile_size);
  c2h::device_vector<flag_t> d_flags(params::tile_size);
  c2h::device_vector<T> d_out(params::tile_size);
  if constexpr (cuda::is_floating_point_v<T>)
  {
    c2h::gen(C2H_SEED(1), d_in, T{-1.0}, T{2.0});
  }
  else
  {
    c2h::gen(C2H_SEED(1), d_in);
  }
  c2h::gen(C2H_SEED(1), d_flags, flag_t{0}, flag_t{2});

  warp_reduce<params::logical_warp_threads, params::total_warps>(
    d_in, d_out, warp_seg_sum_t{thrust::raw_pointer_cast(d_flags.data())});

  // Prepare verification data
  c2h::host_vector<T> h_in         = d_in;
  c2h::host_vector<flag_t> h_flags = d_flags;
  c2h::host_vector<T> h_out        = h_in;
  constexpr auto valid_items       = params::logical_warp_threads;
  compute_host_reference(
    segmented_mod,
    h_in,
    h_flags,
    params::total_warps,
    params::logical_warp_threads,
    valid_items,
    ::cuda::std::plus<T>{},
    h_out.begin());
  verify_results(h_out, d_out);
}

C2H_TEST("WarpReduce::Segmented::Generic",
         "[reduce][warp][segmented]",
         builtin_type_list,
         logical_warp_threads,
         segmented_modes)
{
  using params                 = params_t<TestType>;
  using type                   = typename params::type;
  using red_op_t               = ::cuda::minimum<>;
  constexpr auto segmented_mod = c2h::get<2, TestType>::value;
  using warp_seg_reduction_t =
    cuda::std::_If<(segmented_mod == reduce_mode::tail_flags),
                   warp_seg_reduce_tail_t<type, red_op_t>,
                   warp_seg_reduce_head_t<type, red_op_t>>;

  c2h::device_vector<type> d_in(params::tile_size);
  c2h::device_vector<flag_t> d_flags(params::tile_size);
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::gen(C2H_SEED(5), d_in);
  c2h::gen(C2H_SEED(5), d_flags, flag_t{0}, flag_t{2});

  warp_reduce<params::logical_warp_threads, params::total_warps>(
    d_in, d_out, warp_seg_reduction_t{thrust::raw_pointer_cast(d_flags.data()), red_op_t{}});

  // Prepare verification data
  c2h::host_vector<type> h_in      = d_in;
  c2h::host_vector<flag_t> h_flags = d_flags;
  c2h::host_vector<type> h_out     = h_in;
  constexpr auto valid_items       = params::logical_warp_threads;
  compute_host_reference(
    segmented_mod,
    h_in,
    h_flags,
    params::total_warps,
    params::logical_warp_threads,
    valid_items,
    red_op_t{},
    h_out.begin());
  verify_results(h_out, d_out);
}
