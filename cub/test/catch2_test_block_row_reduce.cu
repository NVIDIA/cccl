// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_row_reduce.cuh>

#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>
#include <c2h/check_results.cuh>
#include <c2h/operator.cuh>

inline constexpr int warp_size = cub::detail::warp_threads;

enum class BlockRowReduceVariant
{
  SharedBroadcast,
  WarpBroadcast
};

enum class BlockRowReduceMode
{
  Sum,
  Reduce
};

template <int RowsPerBlock, int WarpsPerRow>
struct row_reduce_layout
{
  static constexpr int rows_per_block = RowsPerBlock;
  static constexpr int warps_per_row  = WarpsPerRow;
};

struct affine_value_t
{
  int scale;
  int offset;

  friend _CCCL_HOST_DEVICE bool operator==(const affine_value_t& lhs, const affine_value_t& rhs)
  {
    return lhs.scale == rhs.scale && lhs.offset == rhs.offset;
  }
};

struct affine_compose_op
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE affine_value_t operator()(affine_value_t lhs, affine_value_t rhs) const
  {
    constexpr int modulo = 32749;
    return affine_value_t{(lhs.scale * rhs.scale) % modulo, (lhs.scale * rhs.offset + lhs.offset) % modulo};
  }
};

template <>
inline constexpr affine_value_t identity_v<affine_compose_op, affine_value_t> = affine_value_t{1, 0};

struct modular_multiply_op
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static int normalize(int value)
  {
    constexpr int modulo = 251;
    const int result     = value % modulo;
    return result < 0 ? result + modulo : result;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int operator()(int lhs, int rhs) const
  {
    constexpr int modulo = 251;
    return (normalize(lhs) * normalize(rhs)) % modulo;
  }
};

template <>
inline constexpr int identity_v<modular_multiply_op, int> = 1;

template <BlockRowReduceVariant Variant,
          BlockRowReduceMode Mode,
          int RowsPerBlock,
          int WarpsPerRow,
          typename T,
          typename ReductionOp>
__launch_bounds__(RowsPerBlock* WarpsPerRow* warp_size) __global__
  void block_row_reduce_kernel(const T* input, T* output, ReductionOp reduction_op, T identity)
{
  constexpr int block_threads = RowsPerBlock * WarpsPerRow * warp_size;
  static_assert(block_threads <= 1024, "BlockRowReduce test block must fit in a CUDA CTA");

  const int tid = static_cast<int>(threadIdx.x);

  if constexpr (Variant == BlockRowReduceVariant::SharedBroadcast)
  {
    using row_reduce_t = cub::BlockRowReduce<T, RowsPerBlock, WarpsPerRow>;
    __shared__ typename row_reduce_t::TempStorage temp_storage;

    if constexpr (Mode == BlockRowReduceMode::Sum)
    {
      output[tid] = row_reduce_t{temp_storage}.Sum(input[tid]);
    }
    else
    {
      output[tid] = row_reduce_t{temp_storage}.Reduce(input[tid], reduction_op);
    }
  }
  else
  {
    using row_reduce_t = cub::BlockRowReduceWarpBroadcast<T, RowsPerBlock, WarpsPerRow>;
    __shared__ typename row_reduce_t::TempStorage temp_storage;

    if constexpr (Mode == BlockRowReduceMode::Sum)
    {
      output[tid] = row_reduce_t{temp_storage}.Sum(input[tid]);
    }
    else
    {
      output[tid] = row_reduce_t{temp_storage}.CommutativeReduce(input[tid], reduction_op, identity);
    }
  }
}

template <typename T, int MaxReductionItems>
void gen_bounded_input(c2h::seed_t seed, c2h::device_vector<T>& input)
{
  if constexpr (::cuda::std::is_floating_point_v<T>)
  {
    c2h::gen(seed, input, T(0.5), T(1.5));
  }
  else if constexpr (::cuda::std::is_integral_v<T> && ::cuda::std::is_signed_v<T>)
  {
    const T gen_max = ::cuda::std::numeric_limits<T>::max() / static_cast<T>(MaxReductionItems);
    c2h::gen(seed, input, -gen_max, gen_max);
  }
  else if constexpr (::cuda::std::is_integral_v<T>)
  {
    const T gen_max = ::cuda::std::numeric_limits<T>::max() / static_cast<T>(MaxReductionItems);
    c2h::gen(seed, input, T(0), gen_max);
  }
  else
  {
    static_assert(::cuda::std::is_integral_v<T> || ::cuda::std::is_floating_point_v<T>,
                  "gen_bounded_input only supports integral and floating-point types");
  }
}

template <int RowsPerBlock, int WarpsPerRow, typename T, typename ReductionOp>
void compute_host_reference(const c2h::host_vector<T>& input, c2h::host_vector<T>& reference, ReductionOp reduction_op)
{
  static_assert(RowsPerBlock * WarpsPerRow * warp_size <= 1024, "BlockRowReduce test block must fit in a CUDA CTA");

  constexpr int row_threads = WarpsPerRow * warp_size;

  for (int row = 0; row < RowsPerBlock; ++row)
  {
    T aggregate = identity_v<ReductionOp, T>;
    for (int item = 0; item < row_threads; ++item)
    {
      aggregate = static_cast<T>(reduction_op(aggregate, input[row * row_threads + item]));
    }

    for (int item = 0; item < row_threads; ++item)
    {
      reference[row * row_threads + item] = aggregate;
    }
  }
}

template <BlockRowReduceVariant Variant,
          BlockRowReduceMode Mode,
          int RowsPerBlock,
          int WarpsPerRow,
          typename T,
          typename ReductionOp>
void test_block_row_reduce(ReductionOp reduction_op = ReductionOp{})
{
  constexpr int row_threads   = WarpsPerRow * warp_size;
  constexpr int block_threads = RowsPerBlock * row_threads;

  CAPTURE(c2h::type_name<T>(), c2h::type_name<ReductionOp>(), RowsPerBlock, WarpsPerRow, block_threads);

  c2h::device_vector<T> d_input(block_threads);
  gen_bounded_input<T, row_threads>(C2H_SEED(10), d_input);

  c2h::device_vector<T> d_output(block_threads);
  block_row_reduce_kernel<Variant, Mode, RowsPerBlock, WarpsPerRow><<<1, block_threads>>>(
    thrust::raw_pointer_cast(d_input.data()),
    thrust::raw_pointer_cast(d_output.data()),
    reduction_op,
    identity_v<ReductionOp, T>);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<T> h_input(d_input);
  c2h::host_vector<T> h_output(d_output);
  c2h::host_vector<T> h_reference(block_threads);

  compute_host_reference<RowsPerBlock, WarpsPerRow>(h_input, h_reference, reduction_op);
  verify_results(h_reference, h_output);

  for (int row = 0; row < RowsPerBlock; ++row)
  {
    const int row_begin = row * row_threads;
    for (int item = 1; item < row_threads; ++item)
    {
      REQUIRE(h_output[row_begin + item] == h_output[row_begin]);
    }
  }
}

static affine_value_t make_affine_value(int idx)
{
  return affine_value_t{idx % 3 + 1, (idx * 7 + 5) % 17};
}

template <int RowsPerBlock, int WarpsPerRow>
void test_block_row_reduce_non_commutative()
{
  constexpr int row_threads   = WarpsPerRow * warp_size;
  constexpr int block_threads = RowsPerBlock * row_threads;

  c2h::host_vector<affine_value_t> h_input(block_threads);
  for (int idx = 0; idx < static_cast<int>(h_input.size()); ++idx)
  {
    h_input[idx] = make_affine_value(idx);
  }

  c2h::device_vector<affine_value_t> d_input = h_input;
  c2h::device_vector<affine_value_t> d_output(block_threads);
  block_row_reduce_kernel<BlockRowReduceVariant::SharedBroadcast, BlockRowReduceMode::Reduce, RowsPerBlock, WarpsPerRow>
    <<<1, block_threads>>>(
      thrust::raw_pointer_cast(d_input.data()),
      thrust::raw_pointer_cast(d_output.data()),
      affine_compose_op{},
      identity_v<affine_compose_op, affine_value_t>);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<affine_value_t> h_output(d_output);
  c2h::host_vector<affine_value_t> h_reference(block_threads);

  compute_host_reference<RowsPerBlock, WarpsPerRow>(h_input, h_reference, affine_compose_op{});
  REQUIRE(h_reference == h_output);
}

template <BlockRowReduceVariant Variant, int RowsPerBlock, int WarpsPerRow>
__launch_bounds__(RowsPerBlock* WarpsPerRow* warp_size) __global__
  void block_row_reduce_private_storage_kernel(int* output)
{
  constexpr int block_threads = RowsPerBlock * WarpsPerRow * warp_size;
  static_assert(block_threads <= 1024, "BlockRowReduce test block must fit in a CUDA CTA");

  const int tid = static_cast<int>(threadIdx.x);

  if constexpr (Variant == BlockRowReduceVariant::SharedBroadcast)
  {
    output[tid] = cub::BlockRowReduce<int, RowsPerBlock, WarpsPerRow>{}.Sum(tid);
  }
  else
  {
    output[tid] = cub::BlockRowReduceWarpBroadcast<int, RowsPerBlock, WarpsPerRow>{}.Sum(tid);
  }
}

template <BlockRowReduceVariant Variant, int RowsPerBlock, int WarpsPerRow>
void test_block_row_reduce_private_storage()
{
  constexpr int row_threads   = WarpsPerRow * warp_size;
  constexpr int block_threads = RowsPerBlock * row_threads;
  c2h::device_vector<int> d_output(block_threads);

  block_row_reduce_private_storage_kernel<Variant, RowsPerBlock, WarpsPerRow>
    <<<1, block_threads>>>(thrust::raw_pointer_cast(d_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> h_reference(block_threads);
  for (int row = 0; row < RowsPerBlock; ++row)
  {
    const int row_begin = row * row_threads;
    const int row_sum   = (row_begin + row_begin + row_threads - 1) * row_threads / 2;
    for (int item = 0; item < row_threads; ++item)
    {
      h_reference[row_begin + item] = row_sum;
    }
  }

  c2h::host_vector<int> h_output = d_output;
  REQUIRE(h_reference == h_output);
}

template <BlockRowReduceVariant Variant, int RowsPerBlock, int WarpsPerRow>
__launch_bounds__(RowsPerBlock* WarpsPerRow* warp_size) __global__
  void block_row_reduce_multidimensional_kernel(int* output)
{
  constexpr int block_threads = RowsPerBlock * WarpsPerRow * warp_size;
  static_assert(block_threads <= 1024, "BlockRowReduce test block must fit in a CUDA CTA");

  const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

  if constexpr (Variant == BlockRowReduceVariant::SharedBroadcast)
  {
    using row_reduce_t = cub::BlockRowReduce<int, RowsPerBlock, WarpsPerRow>;
    __shared__ typename row_reduce_t::TempStorage temp_storage;
    output[tid] = row_reduce_t{temp_storage}.Sum(tid);
  }
  else
  {
    using row_reduce_t = cub::BlockRowReduceWarpBroadcast<int, RowsPerBlock, WarpsPerRow>;
    __shared__ typename row_reduce_t::TempStorage temp_storage;
    output[tid] = row_reduce_t{temp_storage}.Sum(tid);
  }
}

template <BlockRowReduceVariant Variant, int RowsPerBlock, int WarpsPerRow>
void test_block_row_reduce_multidimensional()
{
  constexpr int row_threads   = WarpsPerRow * warp_size;
  constexpr int block_threads = RowsPerBlock * row_threads;
  static_assert(block_threads % 16 == 0, "BlockRowReduce multidimensional test assumes a 16-wide block dimension");
  c2h::device_vector<int> d_output(block_threads);

  block_row_reduce_multidimensional_kernel<Variant, RowsPerBlock, WarpsPerRow>
    <<<1, dim3(16, block_threads / 16)>>>(thrust::raw_pointer_cast(d_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> h_reference(block_threads);
  for (int row = 0; row < RowsPerBlock; ++row)
  {
    const int row_begin = row * row_threads;
    const int row_sum   = (row_begin + row_begin + row_threads - 1) * row_threads / 2;
    for (int item = 0; item < row_threads; ++item)
    {
      h_reference[row_begin + item] = row_sum;
    }
  }

  c2h::host_vector<int> h_output = d_output;
  REQUIRE(h_reference == h_output);
}

template <BlockRowReduceVariant Variant, int RowsPerBlock, int WarpsPerRow>
__launch_bounds__(RowsPerBlock* WarpsPerRow* warp_size) __global__
  void block_row_reduce_reuse_temp_storage_kernel(int* output)
{
  constexpr int block_threads = RowsPerBlock * WarpsPerRow * warp_size;
  static_assert(block_threads <= 1024, "BlockRowReduce test block must fit in a CUDA CTA");

  const int tid = static_cast<int>(threadIdx.x);

  if constexpr (Variant == BlockRowReduceVariant::SharedBroadcast)
  {
    using row_reduce_t = cub::BlockRowReduce<int, RowsPerBlock, WarpsPerRow>;
    __shared__ typename row_reduce_t::TempStorage temp_storage;
    row_reduce_t row_reduce{temp_storage};

    output[tid] = row_reduce.Sum(tid);
    // Reusing a collective's temporary storage across calls requires a block barrier between calls.
    __syncthreads();
    output[block_threads + tid] = row_reduce.Sum(tid + block_threads);
  }
  else
  {
    using row_reduce_t = cub::BlockRowReduceWarpBroadcast<int, RowsPerBlock, WarpsPerRow>;
    __shared__ typename row_reduce_t::TempStorage temp_storage;
    row_reduce_t row_reduce{temp_storage};

    output[tid] = row_reduce.Sum(tid);
    // Reusing a collective's temporary storage across calls requires a block barrier between calls.
    __syncthreads();
    output[block_threads + tid] = row_reduce.Sum(tid + block_threads);
  }
}

template <int RowsPerBlock, int WarpsPerRow>
__launch_bounds__(RowsPerBlock* WarpsPerRow* warp_size) __global__
  void block_row_reduce_warp_broadcast_float_consistency_kernel(float* output)
{
  constexpr int block_threads = RowsPerBlock * WarpsPerRow * warp_size;
  static_assert(block_threads <= 1024, "BlockRowReduce test block must fit in a CUDA CTA");

  using row_reduce_t = cub::BlockRowReduceWarpBroadcast<float, RowsPerBlock, WarpsPerRow>;
  __shared__ typename row_reduce_t::TempStorage temp_storage;

  const int tid     = static_cast<int>(threadIdx.x);
  const int lane_id = tid % warp_size;
  const int warp_id = tid / warp_size;
  const float input = lane_id % 4 == 0 ? 1.0e20f : (lane_id % 4 == 2 ? -1.0e20f : 1.0f);
  output[tid]       = row_reduce_t{temp_storage}.Sum(input + static_cast<float>(warp_id % WarpsPerRow));
}

template <int RowsPerBlock, int WarpsPerRow>
void test_block_row_reduce_warp_broadcast_float_consistency()
{
  constexpr int row_threads   = WarpsPerRow * warp_size;
  constexpr int block_threads = RowsPerBlock * row_threads;

  c2h::device_vector<float> d_output(block_threads);
  block_row_reduce_warp_broadcast_float_consistency_kernel<RowsPerBlock, WarpsPerRow>
    <<<1, block_threads>>>(thrust::raw_pointer_cast(d_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<float> h_output = d_output;
  for (int row = 0; row < RowsPerBlock; ++row)
  {
    const int row_begin = row * row_threads;
    for (int item = 1; item < row_threads; ++item)
    {
      REQUIRE(h_output[row_begin + item] == h_output[row_begin]);
    }
  }
}

template <BlockRowReduceVariant Variant, int RowsPerBlock, int WarpsPerRow>
void test_block_row_reduce_reuse_temp_storage()
{
  constexpr int row_threads   = WarpsPerRow * warp_size;
  constexpr int block_threads = RowsPerBlock * row_threads;
  c2h::device_vector<int> d_output(2 * block_threads);

  block_row_reduce_reuse_temp_storage_kernel<Variant, RowsPerBlock, WarpsPerRow>
    <<<1, block_threads>>>(thrust::raw_pointer_cast(d_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> h_reference(2 * block_threads);
  for (int pass = 0; pass < 2; ++pass)
  {
    const int input_offset  = pass * block_threads;
    const int output_offset = pass * block_threads;
    for (int row = 0; row < RowsPerBlock; ++row)
    {
      const int row_begin = row * row_threads;
      int row_sum         = 0;
      for (int item = 0; item < row_threads; ++item)
      {
        row_sum += input_offset + row_begin + item;
      }
      for (int item = 0; item < row_threads; ++item)
      {
        h_reference[output_offset + row_begin + item] = row_sum;
      }
    }
  }

  c2h::host_vector<int> h_output = d_output;
  REQUIRE(h_reference == h_output);
}

using value_types = c2h::type_list<::cuda::std::uint16_t, ::cuda::std::int32_t, ::cuda::std::int64_t, float>;
using row_layouts = c2h::type_list<
  row_reduce_layout<1, 1>,
  row_reduce_layout<1, 2>,
  row_reduce_layout<1, 4>,
  row_reduce_layout<1, 8>,
  row_reduce_layout<1, 16>,
  row_reduce_layout<2, 1>,
  row_reduce_layout<2, 2>,
  row_reduce_layout<2, 4>,
  row_reduce_layout<2, 8>,
  row_reduce_layout<2, 16>,
  row_reduce_layout<4, 1>,
  row_reduce_layout<4, 2>,
  row_reduce_layout<4, 4>,
  row_reduce_layout<4, 8>>;
using reduction_ops = c2h::type_list<::cuda::std::plus<>, ::cuda::maximum<>, ::cuda::minimum<>>;

C2H_TEST("BlockRowReduce returns one aggregate per row", "[block][reduce][row]", value_types, row_layouts)
{
  using value_t              = typename c2h::get<0, TestType>;
  using layout_t             = typename c2h::get<1, TestType>;
  constexpr int num_rows     = layout_t::rows_per_block;
  constexpr int warps_in_row = layout_t::warps_per_row;

  test_block_row_reduce<BlockRowReduceVariant::SharedBroadcast, BlockRowReduceMode::Sum, num_rows, warps_in_row, value_t>(
    ::cuda::std::plus<>{});
}

C2H_TEST(
  "BlockRowReduce supports custom row reductions", "[block][reduce][row]", value_types, row_layouts, reduction_ops)
{
  using value_t              = typename c2h::get<0, TestType>;
  using layout_t             = typename c2h::get<1, TestType>;
  constexpr int num_rows     = layout_t::rows_per_block;
  constexpr int warps_in_row = layout_t::warps_per_row;
  using op_t                 = typename c2h::get<2, TestType>;

  test_block_row_reduce<BlockRowReduceVariant::SharedBroadcast,
                        BlockRowReduceMode::Reduce,
                        num_rows,
                        warps_in_row,
                        value_t,
                        op_t>();
}

C2H_TEST(
  "BlockRowReduceWarpBroadcast::Sum returns one aggregate per row", "[block][reduce][row]", value_types, row_layouts)
{
  using value_t              = typename c2h::get<0, TestType>;
  using layout_t             = typename c2h::get<1, TestType>;
  constexpr int num_rows     = layout_t::rows_per_block;
  constexpr int warps_in_row = layout_t::warps_per_row;

  test_block_row_reduce<BlockRowReduceVariant::WarpBroadcast, BlockRowReduceMode::Sum, num_rows, warps_in_row, value_t>(
    ::cuda::std::plus<>{});
}

C2H_TEST("BlockRowReduceWarpBroadcast supports custom row reductions",
         "[block][reduce][row]",
         value_types,
         row_layouts,
         reduction_ops)
{
  using value_t              = typename c2h::get<0, TestType>;
  using layout_t             = typename c2h::get<1, TestType>;
  constexpr int num_rows     = layout_t::rows_per_block;
  constexpr int warps_in_row = layout_t::warps_per_row;
  using op_t                 = typename c2h::get<2, TestType>;

  test_block_row_reduce<BlockRowReduceVariant::WarpBroadcast,
                        BlockRowReduceMode::Reduce,
                        num_rows,
                        warps_in_row,
                        value_t,
                        op_t>();
}

C2H_TEST("BlockRowReduce variants support custom identities", "[block][reduce][row]")
{
  test_block_row_reduce<BlockRowReduceVariant::SharedBroadcast,
                        BlockRowReduceMode::Reduce,
                        1,
                        4,
                        int,
                        modular_multiply_op>();
  test_block_row_reduce<BlockRowReduceVariant::SharedBroadcast,
                        BlockRowReduceMode::Reduce,
                        3,
                        3,
                        int,
                        modular_multiply_op>();
  test_block_row_reduce<BlockRowReduceVariant::WarpBroadcast, BlockRowReduceMode::Reduce, 1, 4, int, modular_multiply_op>();
  test_block_row_reduce<BlockRowReduceVariant::WarpBroadcast, BlockRowReduceMode::Reduce, 3, 3, int, modular_multiply_op>();
}

C2H_TEST("BlockRowReduce supports associative non-commutative row reductions", "[block][reduce][row]")
{
  test_block_row_reduce_non_commutative<1, 1>();
  test_block_row_reduce_non_commutative<1, 4>();
  test_block_row_reduce_non_commutative<1, 8>();
  test_block_row_reduce_non_commutative<2, 2>();
  test_block_row_reduce_non_commutative<2, 8>();
}

C2H_TEST("BlockRowReduce supports private temporary storage", "[block][reduce][row]")
{
  test_block_row_reduce_private_storage<BlockRowReduceVariant::SharedBroadcast, 1, 1>();
  test_block_row_reduce_private_storage<BlockRowReduceVariant::SharedBroadcast, 2, 2>();
  test_block_row_reduce_private_storage<BlockRowReduceVariant::WarpBroadcast, 1, 1>();
  test_block_row_reduce_private_storage<BlockRowReduceVariant::WarpBroadcast, 2, 2>();
}

C2H_TEST("BlockRowReduce supports multidimensional thread blocks", "[block][reduce][row]")
{
  test_block_row_reduce_multidimensional<BlockRowReduceVariant::SharedBroadcast, 2, 2>();
  test_block_row_reduce_multidimensional<BlockRowReduceVariant::WarpBroadcast, 2, 2>();
}

C2H_TEST("BlockRowReduce supports temporary storage reuse", "[block][reduce][row]")
{
  test_block_row_reduce_reuse_temp_storage<BlockRowReduceVariant::SharedBroadcast, 1, 4>();
  test_block_row_reduce_reuse_temp_storage<BlockRowReduceVariant::SharedBroadcast, 2, 8>();
  test_block_row_reduce_reuse_temp_storage<BlockRowReduceVariant::WarpBroadcast, 1, 4>();
  test_block_row_reduce_reuse_temp_storage<BlockRowReduceVariant::WarpBroadcast, 2, 8>();
}

C2H_TEST("BlockRowReduceWarpBroadcast returns lane-invariant floating-point sums", "[block][reduce][row]")
{
  test_block_row_reduce_warp_broadcast_float_consistency<1, 1>();
  test_block_row_reduce_warp_broadcast_float_consistency<1, 4>();
  test_block_row_reduce_warp_broadcast_float_consistency<2, 3>();
}

C2H_TEST("BlockRowReduce supports maximum block width", "[block][reduce][row]")
{
  test_block_row_reduce<BlockRowReduceVariant::SharedBroadcast, BlockRowReduceMode::Sum, 1, 32, ::cuda::std::int32_t>(
    ::cuda::std::plus<>{});
  test_block_row_reduce<BlockRowReduceVariant::SharedBroadcast, BlockRowReduceMode::Sum, 32, 1, ::cuda::std::int32_t>(
    ::cuda::std::plus<>{});
  test_block_row_reduce<BlockRowReduceVariant::WarpBroadcast, BlockRowReduceMode::Sum, 1, 32, ::cuda::std::int32_t>(
    ::cuda::std::plus<>{});
  test_block_row_reduce<BlockRowReduceVariant::WarpBroadcast, BlockRowReduceMode::Sum, 32, 1, ::cuda::std::int32_t>(
    ::cuda::std::plus<>{});
}

C2H_TEST("BlockRowReduce supports non-power-of-two row layouts", "[block][reduce][row]")
{
  test_block_row_reduce<BlockRowReduceVariant::SharedBroadcast, BlockRowReduceMode::Sum, 3, 3, ::cuda::std::int32_t>(
    ::cuda::std::plus<>{});
  test_block_row_reduce<BlockRowReduceVariant::SharedBroadcast, BlockRowReduceMode::Sum, 5, 3, ::cuda::std::int32_t>(
    ::cuda::std::plus<>{});
  test_block_row_reduce<BlockRowReduceVariant::WarpBroadcast, BlockRowReduceMode::Sum, 3, 3, ::cuda::std::int32_t>(
    ::cuda::std::plus<>{});
  test_block_row_reduce<BlockRowReduceVariant::WarpBroadcast, BlockRowReduceMode::Sum, 5, 3, ::cuda::std::int32_t>(
    ::cuda::std::plus<>{});
}
