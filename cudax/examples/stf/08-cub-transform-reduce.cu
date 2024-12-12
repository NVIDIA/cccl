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
 * @brief Example of reduction implementing using CUB
 */

#include <cub/cub.cuh>

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

/**
 * This functor transforms a 1D index into the result of the transformation
 */
// Args... is for example slice<double>, slice<int>...
template <typename TransformOp, typename shape_t, typename... Args>
struct IndexToTransformedValue
{
  IndexToTransformedValue(TransformOp _op, shape_t s, ::std::tuple<Args...> _targs)
      : op(mv(_op))
      , shape(mv(s))
      , targs(mv(_targs))
  {}

  __host__ __device__ __forceinline__ auto operator()(const size_t& index) const
  {
    const auto explode_args = [&](auto&&... data) {
      CUDASTF_NO_DEVICE_STACK
      auto const explode_coords = [&](auto&&... coords) {
        return op(coords..., data...);
      };
      return ::std::apply(explode_coords, shape.index_to_coords(index));
    };
    return ::std::apply(explode_args, targs);
  }

  TransformOp op;
  const shape_t shape;
  ::std::tuple<Args...> targs;
};

/**
 * @brief Helper to transform a device lambda into a functor that we can use in CUB
 */
template <typename BinaryOp>
struct ReduceOpWrapper
{
  ReduceOpWrapper(BinaryOp _op)
      : op(mv(_op)) {};

  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return op(a, b);
  }

  BinaryOp op;
};

// This should print the output of the transform op (which is an int)
template <typename It>
__global__ void TEST_KERNEL(It it)
{
  printf("it(%d) = %d\n", threadIdx.x, it[threadIdx.x]);
}

/**
 * @brief Remove the first entry of a std::tuple
 */
template <typename Tuple>
auto remove_first(const Tuple& t)
{
  return ::std::apply(
    [](auto&& head, auto&&... tail) {
      return ::std::make_tuple(::std::forward<decltype(tail)>(tail)...);
    },
    t);
}

template <typename Ctx, typename shape_t, typename TransformOp, typename BinaryOp, typename OutT, typename... Args>
auto stf_transform_reduce(
  Ctx& ctx, shape_t s, TransformOp&& transform_op, BinaryOp&& op, OutT init_val, logical_data<Args>... args)
{
  using ConvertionOp_t = IndexToTransformedValue<TransformOp, shape_t, Args...>;

  // The result of this operation is a logical data
  auto result = ctx.logical_data(shape_of<scalar<OutT>>());

  auto t = ctx.task(result.write(), args.read()...);
  t.start();
  cudaStream_t stream = t.get_stream();

  auto deps = t.typed_deps();
  // We remove the first argument
  ConvertionOp_t conversion_op(transform_op, s, remove_first(deps));

  size_t num_elements = s.size();
  cub::CountingInputIterator<size_t> count_it(0);

  // Create an iterator wrapper
  cub::TransformInputIterator<OutT, ConvertionOp_t, decltype(count_it)> itr(count_it, conversion_op);

  // Ensure that the
  TEST_KERNEL<<<1, 8, 0, stream>>>(itr);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(
    d_temp_storage,
    temp_storage_bytes,
    itr, //*static_cast<decltype(itr) *>(nullptr), // TODO
    (OutT*) nullptr,
    num_elements,
    ReduceOpWrapper<BinaryOp>(op),
    init_val,
    0);

  cuda_safe_call(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

  cub::DeviceReduce::Reduce(
    d_temp_storage,
    temp_storage_bytes,
    itr, // TODO
    (OutT*) ::std::get<0>(deps).addr,
    num_elements,
    ReduceOpWrapper<BinaryOp>(op),
    init_val,
    0);

  cuda_safe_call(cudaFreeAsync(d_temp_storage, stream));

  t.end();

  return result;
}

template <typename Ctx>
void run()
{
  Ctx ctx;

  const size_t N = 1024 * 16;

  int ref_prod = 0;

  int* X = new int[N];
  int* Y = new int[N];

  for (int ind = 0; ind < N; ind++)
  {
    X[ind] = 2 + ind; // rand() % N;
    Y[ind] = 3 + ind; // rand() % N;
    ref_prod += X[ind] * Y[ind];
  }

  auto lX = ctx.logical_data(X, {N});
  auto lY = ctx.logical_data(Y, {N});

  auto lresult = stf_transform_reduce(
    ctx,
    lX.shape(),
    [] __device__(size_t i, auto x, auto y) {
      return x(i) * y(i);
    },
    [] __device__(const int& a, const int& b) {
      return a + b;
    },
    0,
    lX,
    lY);

  int result = ctx.wait(lresult);
  _CCCL_ASSERT(result == ref_prod, "Incorrect result");

  ctx.finalize();
}

int main()
{
  run<stream_ctx>();
  // run<graph_ctx>();
}
