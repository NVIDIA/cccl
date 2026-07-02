// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_reduce_broadcast.cuh>

#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>
#include <c2h/check_results.cuh>
#include <c2h/operator.cuh>

inline constexpr int num_items_per_thread = 4;

enum class BlockReduceBroadcastMode
{
  Sum,
  SumValidItems,
  SumMultipleItems,
  Reduce,
  ReduceValidItems,
  ReduceMultipleItems
};

template <BlockReduceBroadcastMode Mode>
inline constexpr int items_per_thread_for_mode_v =
  (Mode == BlockReduceBroadcastMode::SumMultipleItems || Mode == BlockReduceBroadcastMode::ReduceMultipleItems)
    ? num_items_per_thread
    : 1;

template <BlockReduceBroadcastMode Mode, int BlockDimX, int BlockDimY, int BlockDimZ>
inline constexpr int max_reduction_items_v = BlockDimX * BlockDimY * BlockDimZ * items_per_thread_for_mode_v<Mode>;

template <typename ReductionOp>
inline constexpr bool is_commutative_test_op_v =
  ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<ReductionOp>, ::cuda::std::plus<>>
  || ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<ReductionOp>, ::cuda::maximum<>>
  || ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<ReductionOp>, ::cuda::minimum<>>;

struct affine_value_t
{
  int scale;
  int offset;

  friend bool operator==(const affine_value_t& lhs, const affine_value_t& rhs)
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

struct non_trivial_value_t
{
  int value;

  _CCCL_HOST_DEVICE non_trivial_value_t()
      : value(0)
  {}

  _CCCL_HOST_DEVICE explicit non_trivial_value_t(int value)
      : value(value)
  {}

  // Keep these copy operations user-provided so this remains non-trivially copyable.
  _CCCL_HOST_DEVICE non_trivial_value_t(const non_trivial_value_t& other) // NOLINT(modernize-use-equals-default)
      : value(other.value)
  {}

  _CCCL_HOST_DEVICE non_trivial_value_t&
  operator=(const non_trivial_value_t& other) // NOLINT(modernize-use-equals-default)
  {
    value = other.value;
    return *this;
  }

  friend _CCCL_HOST_DEVICE bool operator==(non_trivial_value_t lhs, non_trivial_value_t rhs)
  {
    return lhs.value == rhs.value;
  }
};

static_assert(!::cuda::std::is_trivially_copyable_v<non_trivial_value_t>);
static_assert(::cuda::std::is_trivially_destructible_v<non_trivial_value_t>);
static_assert(!thrust::is_trivially_relocatable_v<non_trivial_value_t>);

struct non_trivial_sum_op
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE non_trivial_value_t
  operator()(non_trivial_value_t lhs, non_trivial_value_t rhs) const
  {
    return non_trivial_value_t{lhs.value + rhs.value};
  }
};

template <BlockReduceBroadcastMode Mode,
          cub::BlockReduceAlgorithm Algorithm,
          int ItemsPerThread,
          int BlockDimX,
          int BlockDimY,
          int BlockDimZ,
          typename T,
          typename ReductionOp>
__launch_bounds__(BlockDimX* BlockDimY* BlockDimZ) __global__
  void block_reduce_broadcast_kernel(const T* input, T* output, ReductionOp reduction_op, int valid_items)
{
  constexpr int block_threads = BlockDimX * BlockDimY * BlockDimZ;
  static_assert(block_threads > 0, "Block must contain at least one thread");

  using block_reduce_t = cub::BlockReduceBroadcast<T, BlockDimX, Algorithm, BlockDimY, BlockDimZ>;

  __shared__ typename block_reduce_t::TempStorage temp_storage;

  const int tid           = static_cast<int>(cub::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ));
  const int thread_offset = tid * ItemsPerThread;
  T thread_data[ItemsPerThread];

  // The test buffer is intentionally sized for the full tile, including lanes masked by valid_items overloads.
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    thread_data[item] = input[thread_offset + item];
  }

  if constexpr (Mode == BlockReduceBroadcastMode::Sum)
  {
    output[tid] = block_reduce_t{temp_storage}.Sum(thread_data[0]);
  }
  else if constexpr (Mode == BlockReduceBroadcastMode::SumValidItems)
  {
    output[tid] = block_reduce_t{temp_storage}.Sum(thread_data[0], valid_items);
  }
  else if constexpr (Mode == BlockReduceBroadcastMode::SumMultipleItems)
  {
    output[tid] = block_reduce_t{temp_storage}.Sum(thread_data);
  }
  else if constexpr (Mode == BlockReduceBroadcastMode::Reduce)
  {
    output[tid] = block_reduce_t{temp_storage}.Reduce(thread_data[0], reduction_op);
  }
  else if constexpr (Mode == BlockReduceBroadcastMode::ReduceValidItems)
  {
    output[tid] = block_reduce_t{temp_storage}.Reduce(thread_data[0], reduction_op, valid_items);
  }
  else
  {
    output[tid] = block_reduce_t{temp_storage}.Reduce(thread_data, reduction_op);
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
    using bound_t      = long long;
    const auto raw_max = static_cast<bound_t>(::cuda::std::numeric_limits<T>::max()) / MaxReductionItems;
    const T gen_max    = static_cast<T>(raw_max > bound_t{0} ? raw_max : bound_t{1});
    c2h::gen(seed, input, -gen_max, gen_max);
  }
  else if constexpr (::cuda::std::is_integral_v<T>)
  {
    using bound_t           = unsigned long long;
    constexpr auto raw_max  = static_cast<bound_t>(::cuda::std::numeric_limits<T>::max()) / MaxReductionItems;
    constexpr bool is_bound = raw_max > bound_t{0};
    if constexpr (is_bound)
    {
      c2h::gen(seed, input, T(0), static_cast<T>(raw_max));
    }
    else
    {
      // Exercise the full unsigned range when no positive bounded range exists. Small unsigned Sum cases deliberately
      // test modular arithmetic; the device result and host reference use the same wrapping semantics.
      c2h::gen(seed, input);
    }
  }
  else
  {
    c2h::gen(seed, input);
  }
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244) // numeric(33): C: '=': conversion from 'int' to '_Ty', possible loss of data

template <int ItemsPerThread, int BlockThreads, typename T, typename ReductionOp>
void compute_host_reference(
  const c2h::host_vector<T>& input, c2h::host_vector<T>& reference, ReductionOp reduction_op, int valid_items)
{
  static_assert(ItemsPerThread > 0, "ItemsPerThread must be greater than zero");

  auto aggregate = identity_v<ReductionOp, T>;
  for (int thread = 0; thread < BlockThreads; ++thread)
  {
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int idx = thread * ItemsPerThread + item;
      if (idx < valid_items)
      {
        aggregate = reduction_op(aggregate, input[idx]);
      }
    }
  }

  for (int thread = 0; thread < BlockThreads; ++thread)
  {
    reference[thread] = static_cast<T>(aggregate);
  }
}

_CCCL_DIAG_POP

template <int BlockThreads, int ItemsPerThread, bool IsPartial, typename F>
void for_each_valid_items(F callback)
{
  if constexpr (IsPartial)
  {
    // Keep the partial-coverage set unique for small blocks while still exercising first, middle, near-full, and full
    // valid ranges for larger blocks.
    callback(::cuda::std::integral_constant<int, 1>{});
    if constexpr (BlockThreads > 2)
    {
      callback(::cuda::std::integral_constant<int, BlockThreads / 2>{});
      callback(::cuda::std::integral_constant<int, BlockThreads - 1>{});
    }
    if constexpr (BlockThreads > 1)
    {
      callback(::cuda::std::integral_constant<int, BlockThreads>{});
    }
  }
  else
  {
    callback(::cuda::std::integral_constant<int, BlockThreads * ItemsPerThread>{});
  }
}

template <BlockReduceBroadcastMode Mode,
          cub::BlockReduceAlgorithm Algorithm,
          int BlockDimX,
          int BlockDimY,
          int BlockDimZ,
          typename T,
          typename ReductionOp>
void test_block_reduce_broadcast(ReductionOp reduction_op = ReductionOp{})
{
  static_assert(Algorithm != cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
                  || is_commutative_test_op_v<ReductionOp>,
                "BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY test cases require a commutative reduction operator");

  constexpr int items_per_thread = items_per_thread_for_mode_v<Mode>;
  constexpr int block_threads    = BlockDimX * BlockDimY * BlockDimZ;
  constexpr bool is_partial =
    (Mode == BlockReduceBroadcastMode::SumValidItems || Mode == BlockReduceBroadcastMode::ReduceValidItems);

  for_each_valid_items<BlockDimX * BlockDimY * BlockDimZ, items_per_thread_for_mode_v<Mode>, is_partial>(
    [&](auto valid_items_constant) {
      constexpr int valid_items = decltype(valid_items_constant)::value;

      CAPTURE(c2h::type_name<T>(),
              c2h::type_name<ReductionOp>(),
              Algorithm,
              BlockDimX,
              BlockDimY,
              BlockDimZ,
              items_per_thread,
              valid_items);

      c2h::device_vector<T> d_input(block_threads * items_per_thread);
      gen_bounded_input<T, max_reduction_items_v<Mode, BlockDimX, BlockDimY, BlockDimZ>>(C2H_SEED(10), d_input);

      c2h::device_vector<T> d_output(block_threads);
      dim3 block_dims(BlockDimX, BlockDimY, BlockDimZ);
      block_reduce_broadcast_kernel<Mode, Algorithm, items_per_thread_for_mode_v<Mode>, BlockDimX, BlockDimY, BlockDimZ>
        <<<1, block_dims>>>(thrust::raw_pointer_cast(d_input.data()),
                            thrust::raw_pointer_cast(d_output.data()),
                            reduction_op,
                            valid_items);

      REQUIRE(cudaSuccess == cudaPeekAtLastError());
      REQUIRE(cudaSuccess == cudaDeviceSynchronize());

      c2h::host_vector<T> h_input  = d_input;
      c2h::host_vector<T> h_output = d_output;
      c2h::host_vector<T> h_reference(block_threads);

      compute_host_reference<items_per_thread_for_mode_v<Mode>, BlockDimX * BlockDimY * BlockDimZ>(
        h_input, h_reference, reduction_op, valid_items);
      verify_results(h_reference, h_output);
    });
}

static affine_value_t make_affine_value(int idx)
{
  return affine_value_t{idx % 3 + 1, (idx * 7 + 5) % 17};
}

template <BlockReduceBroadcastMode Mode,
          cub::BlockReduceAlgorithm Algorithm,
          int BlockDimX = 64,
          int BlockDimY = 1,
          int BlockDimZ = 1>
void test_block_reduce_broadcast_non_commutative()
{
  static_assert(Mode == BlockReduceBroadcastMode::Reduce || Mode == BlockReduceBroadcastMode::ReduceMultipleItems,
                "Only full-tile Reduce modes are tested with the non-commutative operator");

  constexpr int block_threads    = BlockDimX * BlockDimY * BlockDimZ;
  constexpr int items_per_thread = Mode == BlockReduceBroadcastMode::ReduceMultipleItems ? num_items_per_thread : 1;
  constexpr int valid_items      = block_threads * items_per_thread;

  c2h::host_vector<affine_value_t> h_input(block_threads * items_per_thread);
  for (int idx = 0; idx < static_cast<int>(h_input.size()); ++idx)
  {
    h_input[idx] = make_affine_value(idx);
  }

  c2h::device_vector<affine_value_t> d_input = h_input;
  c2h::device_vector<affine_value_t> d_output(block_threads);
  dim3 block_dims(BlockDimX, BlockDimY, BlockDimZ);
  block_reduce_broadcast_kernel<Mode, Algorithm, items_per_thread, BlockDimX, BlockDimY, BlockDimZ><<<1, block_dims>>>(
    thrust::raw_pointer_cast(d_input.data()),
    thrust::raw_pointer_cast(d_output.data()),
    affine_compose_op{},
    valid_items);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<affine_value_t> h_output = d_output;
  c2h::host_vector<affine_value_t> h_reference(block_threads);

  compute_host_reference<items_per_thread, block_threads>(h_input, h_reference, affine_compose_op{}, valid_items);
  REQUIRE(h_reference == h_output);
}

template <cub::BlockReduceAlgorithm Algorithm>
__global__ void block_reduce_broadcast_reuse_storage_kernel(int* output)
{
  constexpr int block_dim_x = 64;
  using block_reduce_t      = cub::BlockReduceBroadcast<int, block_dim_x, Algorithm>;

  __shared__ typename block_reduce_t::TempStorage temp_storage;

  block_reduce_t block_reduce(temp_storage);
  const auto first  = block_reduce.Sum(static_cast<int>(threadIdx.x));
  const auto second = block_reduce.Sum(static_cast<int>(threadIdx.x + 1));

  output[threadIdx.x]               = first;
  output[threadIdx.x + block_dim_x] = second;
}

template <cub::BlockReduceAlgorithm Algorithm>
void test_block_reduce_broadcast_reuse_storage()
{
  constexpr int block_dim_x = 64;
  c2h::device_vector<int> d_output(block_dim_x * 2);

  block_reduce_broadcast_reuse_storage_kernel<Algorithm><<<1, block_dim_x>>>(thrust::raw_pointer_cast(d_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> h_reference(block_dim_x * 2);
  for (int thread = 0; thread < block_dim_x; ++thread)
  {
    h_reference[thread]               = 2016;
    h_reference[thread + block_dim_x] = 2080;
  }

  REQUIRE(h_reference == d_output);
}

template <cub::BlockReduceAlgorithm Algorithm>
__global__ void block_reduce_broadcast_non_trivial_kernel(non_trivial_value_t* output)
{
  constexpr int block_dim_x = 64;
  using block_reduce_t      = cub::BlockReduceBroadcast<non_trivial_value_t, block_dim_x, Algorithm>;

  __shared__ typename block_reduce_t::TempStorage temp_storage;

  const auto input    = non_trivial_value_t{static_cast<int>(threadIdx.x)};
  const auto result   = block_reduce_t{temp_storage}.Reduce(input, non_trivial_sum_op{});
  output[threadIdx.x] = result;
}

template <cub::BlockReduceAlgorithm Algorithm>
void test_block_reduce_broadcast_non_trivial()
{
  constexpr int block_dim_x = 64;
  c2h::device_vector<non_trivial_value_t> d_output(block_dim_x);

  block_reduce_broadcast_non_trivial_kernel<Algorithm><<<1, block_dim_x>>>(thrust::raw_pointer_cast(d_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<non_trivial_value_t> h_output = d_output;
  for (int thread = 0; thread < block_dim_x; ++thread)
  {
    CAPTURE(thread);
    REQUIRE(h_output[thread].value == 2016);
  }
}

template <cub::BlockReduceAlgorithm Algorithm>
__global__ void block_reduce_broadcast_private_storage_kernel(int* output)
{
  constexpr int block_dim_x = 64;
  using block_reduce_t      = cub::BlockReduceBroadcast<int, block_dim_x, Algorithm>;

  const auto result   = block_reduce_t{}.Sum(static_cast<int>(threadIdx.x));
  output[threadIdx.x] = result;
}

template <cub::BlockReduceAlgorithm Algorithm>
void test_block_reduce_broadcast_private_storage()
{
  constexpr int block_dim_x = 64;
  c2h::device_vector<int> d_output(block_dim_x);

  block_reduce_broadcast_private_storage_kernel<Algorithm>
    <<<1, block_dim_x>>>(thrust::raw_pointer_cast(d_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> h_reference(block_dim_x, 2016);
  REQUIRE(h_reference == d_output);
}

using value_types =
  c2h::type_list<::cuda::std::uint8_t, ::cuda::std::uint16_t, ::cuda::std::int32_t, ::cuda::std::int64_t, float, double>;
using block_dim_xs  = c2h::enum_type_list<int, 64, 128>;
using block_dim_yzs = c2h::enum_type_list<int, 1, 2>;
using commutative_algorithms =
  c2h::enum_type_list<cub::BlockReduceAlgorithm,
                      cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING,
                      cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                      cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>;
using ordered_algorithms =
  c2h::enum_type_list<cub::BlockReduceAlgorithm,
                      cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING,
                      cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>;
using commutative_only_algorithms =
  c2h::enum_type_list<cub::BlockReduceAlgorithm, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
using reduction_ops = c2h::type_list<::cuda::maximum<>, ::cuda::minimum<>>;

C2H_TEST("BlockReduceBroadcast::Sum returns the aggregate to every thread",
         "[block][reduce]",
         value_types,
         block_dim_xs,
         block_dim_yzs,
         block_dim_yzs,
         commutative_algorithms)
{
  using value_t             = typename c2h::get<0, TestType>;
  constexpr int block_dim_x = c2h::get<1, TestType>::value;
  constexpr int block_dim_y = c2h::get<2, TestType>::value;
  constexpr int block_dim_z = c2h::get<3, TestType>::value;
  constexpr auto algorithm  = c2h::get<4, TestType>::value;

  test_block_reduce_broadcast<BlockReduceBroadcastMode::Sum, algorithm, block_dim_x, block_dim_y, block_dim_z, value_t>(
    ::cuda::std::plus<>{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::SumValidItems,
                              algorithm,
                              block_dim_x,
                              block_dim_y,
                              block_dim_z,
                              value_t>(::cuda::std::plus<>{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::SumMultipleItems,
                              algorithm,
                              block_dim_x,
                              block_dim_y,
                              block_dim_z,
                              value_t>(::cuda::std::plus<>{});
}

C2H_TEST("BlockReduceBroadcast::Reduce returns the aggregate to every thread",
         "[block][reduce]",
         value_types,
         block_dim_xs,
         block_dim_yzs,
         block_dim_yzs,
         ordered_algorithms,
         reduction_ops)
{
  using value_t             = typename c2h::get<0, TestType>;
  constexpr int block_dim_x = c2h::get<1, TestType>::value;
  constexpr int block_dim_y = c2h::get<2, TestType>::value;
  constexpr int block_dim_z = c2h::get<3, TestType>::value;
  constexpr auto algorithm  = c2h::get<4, TestType>::value;
  using op_t                = typename c2h::get<5, TestType>;

  test_block_reduce_broadcast<BlockReduceBroadcastMode::Reduce, algorithm, block_dim_x, block_dim_y, block_dim_z, value_t>(
    op_t{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::ReduceValidItems,
                              algorithm,
                              block_dim_x,
                              block_dim_y,
                              block_dim_z,
                              value_t>(op_t{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::ReduceMultipleItems,
                              algorithm,
                              block_dim_x,
                              block_dim_y,
                              block_dim_z,
                              value_t>(op_t{});
}

C2H_TEST("BlockReduceBroadcast::Reduce supports commutative-only algorithms",
         "[block][reduce]",
         value_types,
         commutative_only_algorithms,
         reduction_ops)
{
  using value_t            = typename c2h::get<0, TestType>;
  constexpr auto algorithm = c2h::get<1, TestType>::value;
  using op_t               = typename c2h::get<2, TestType>;

  test_block_reduce_broadcast<BlockReduceBroadcastMode::Reduce, algorithm, 64, 1, 1, value_t>(op_t{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::ReduceValidItems, algorithm, 64, 1, 1, value_t>(op_t{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::ReduceMultipleItems, algorithm, 64, 1, 1, value_t>(op_t{});
}

C2H_TEST("BlockReduceBroadcast::Sum supports nondeterministic warp reductions", "[block][reduce]")
{
  test_block_reduce_broadcast<BlockReduceBroadcastMode::Sum,
                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
                              128,
                              1,
                              1,
                              int>(::cuda::std::plus<>{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::SumMultipleItems,
                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
                              128,
                              1,
                              1,
                              int>(::cuda::std::plus<>{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::Sum,
                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
                              64,
                              1,
                              1,
                              float>(::cuda::std::plus<>{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::SumMultipleItems,
                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
                              64,
                              1,
                              1,
                              float>(::cuda::std::plus<>{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::Sum,
                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
                              256,
                              1,
                              1,
                              double>(::cuda::std::plus<>{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::SumMultipleItems,
                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
                              256,
                              1,
                              1,
                              double>(::cuda::std::plus<>{});
}

C2H_TEST("BlockReduceBroadcast supports irregular block layouts", "[block][reduce]")
{
  test_block_reduce_broadcast<BlockReduceBroadcastMode::Sum, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING, 1, 1, 1, int>(
    ::cuda::std::plus<>{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::SumValidItems,
                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING,
                              7,
                              1,
                              1,
                              int>(::cuda::std::plus<>{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::Reduce,
                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS,
                              65,
                              1,
                              1,
                              int>(::cuda::maximum<>{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::ReduceValidItems,
                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS,
                              13,
                              3,
                              1,
                              int>(::cuda::minimum<>{});
  test_block_reduce_broadcast<BlockReduceBroadcastMode::SumMultipleItems,
                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                              65,
                              1,
                              1,
                              int>(::cuda::std::plus<>{});
}

C2H_TEST("BlockReduceBroadcast::Reduce preserves non-commutative reduction order", "[block][reduce]")
{
  test_block_reduce_broadcast_non_commutative<BlockReduceBroadcastMode::Reduce,
                                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>();
  test_block_reduce_broadcast_non_commutative<BlockReduceBroadcastMode::ReduceMultipleItems,
                                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>();
  test_block_reduce_broadcast_non_commutative<BlockReduceBroadcastMode::Reduce,
                                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>();
  test_block_reduce_broadcast_non_commutative<BlockReduceBroadcastMode::ReduceMultipleItems,
                                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>();
  test_block_reduce_broadcast_non_commutative<BlockReduceBroadcastMode::Reduce,
                                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING,
                                              32,
                                              2,
                                              1>();
  test_block_reduce_broadcast_non_commutative<BlockReduceBroadcastMode::ReduceMultipleItems,
                                              cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING,
                                              32,
                                              2,
                                              1>();
}

C2H_TEST("BlockReduceBroadcast supports non-trivially-copyable value types", "[block][reduce]")
{
  test_block_reduce_broadcast_non_trivial<cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>();
  test_block_reduce_broadcast_non_trivial<cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>();
  test_block_reduce_broadcast_non_trivial<cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>();
}

C2H_TEST("BlockReduceBroadcast supports reusing temporary storage", "[block][reduce]")
{
  test_block_reduce_broadcast_reuse_storage<cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>();
  test_block_reduce_broadcast_reuse_storage<cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>();
  test_block_reduce_broadcast_reuse_storage<cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>();
}

C2H_TEST("BlockReduceBroadcast supports private temporary storage", "[block][reduce]")
{
  test_block_reduce_broadcast_private_storage<cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING>();
  test_block_reduce_broadcast_private_storage<cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>();
  test_block_reduce_broadcast_private_storage<cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS>();
}
