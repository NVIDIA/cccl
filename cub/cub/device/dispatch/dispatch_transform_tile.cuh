// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal dispatch helpers for cub::DeviceTransform's tile path:
//   tile_dispatch_eligible_v  -- compile-time predicate the hook consults
//   runtime_preconditions_ok  -- runtime alignment + divisibility predicate
//   dispatch                  -- bridge that launches the tile kernel with
//                                the trait's substitute functor
//   DeviceTransform           -- internal tile-local Transform/Fill wrappers
//                                used by `dispatch`
// User-facing extension points (tile_eligible / tile_mufu_heavy) live in
// dispatch_transform_tile_traits.cuh under cub::transform.
// Requires CTK 13.3 or newer and nvcc invoked with --enable-tile.

#pragma once

#include <cub/config.cuh>

#include <cub/device/dispatch/dispatch_transform_tile_config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUB_HAS_TILE_TRANSFORM()

#  include <cub/device/dispatch/dispatch_transform_tile_traits.cuh>
#  include <cub/device/dispatch/kernels/kernel_transform_tile.cuh>
#  include <cub/device/dispatch/tuning/tuning_transform_tile.cuh>

#  include <thrust/type_traits/is_contiguous_iterator.h>
#  include <thrust/type_traits/unwrap_contiguous_iterator.h>

#  include <cuda/std/__memory/is_sufficiently_aligned.h>
#  include <cuda/std/__tuple_dir/apply.h>
#  include <cuda/std/__type_traits/is_empty.h>
#  include <cuda/std/__type_traits/is_trivially_default_constructible.h>
#  include <cuda/std/__utility/declval.h>
#  include <cuda/std/__type_traits/remove_cv.h>
#  include <cuda/std/__type_traits/remove_pointer.h>
#  include <cuda/std/tuple>
#  include <cuda/std/utility>

#  include <cuda_runtime.h>

#  include <cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::transform::tile
{

template <int TileSize, typename Fn, typename Out, typename... Ins, ::cuda::std::size_t... Idx>
cudaError_t launch_impl(
  ::cuda::std::tuple<Ins*...> inputs,
  Out* output,
  int64_t num_items,
  cudaStream_t stream,
  ::cuda::std::index_sequence<Idx...>)
{
  if (num_items <= 0)
  {
    return cudaSuccess;
  }

  const int64_t num_blocks = (num_items + TileSize - 1) / TileSize;

  transform_kernel<TileSize, Fn><<<static_cast<unsigned int>(num_blocks), 1, 0, stream>>>(
    num_items, output, ::cuda::std::get<Idx>(inputs)...);

  return cudaGetLastError();
}

struct DeviceTransform
{
  template <int TileSize = 0, bool MufuHeavy = false, typename Fn, typename Out, typename... Ins>
  static cudaError_t
  Transform(::cuda::std::tuple<Ins*...> inputs, Out* output, int64_t num_items, Fn, cudaStream_t stream = 0)
  {
    constexpr int chosen = (TileSize > 0) ? TileSize : pick_tile_size<Out, Ins...>(MufuHeavy);
    return launch_impl<chosen, Fn>(inputs, output, num_items, stream, ::cuda::std::index_sequence_for<Ins...>{});
  }

  // Fill
  template <int TileSize = 0, typename T>
  static cudaError_t Fill(T* output, int64_t num_items, T value, cudaStream_t stream = 0)
  {
    if (num_items <= 0)
    {
      return cudaSuccess;
    }
    constexpr int chosen     = (TileSize > 0) ? TileSize : pick_tile_size<T>();
    const int64_t num_blocks = (num_items + chosen - 1) / chosen;
    fill_kernel<chosen, T><<<static_cast<unsigned int>(num_blocks), 1, 0, stream>>>(num_items, output, value);
    return cudaGetLastError();
  }
};

namespace __detail
{
template <typename Iter>
using __unwrapped_value_t =
  ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<decltype(THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(
    ::cuda::std::declval<Iter>()))>>;
} // namespace __detail

// Combined compile-time predicate used by cub::DeviceTransform's __transform_internal
// to decide whether to route a given (Op, OutIter, InIters...) to the tile path.
// The call site lifts this into an `if constexpr`: when this is true the hook
// tries the tile kernel first and, on runtime alignment / divisibility
// failure, falls through to the standard CUB dispatch below. When false, the
// tile branch is discarded and only CUB's standard path is emitted.
template <typename Op, typename OutIter, typename... InIters>
inline constexpr bool tile_dispatch_eligible_v =
  THRUST_NS_QUALIFIER::is_contiguous_iterator_v<OutIter>
  && (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<InIters> && ...)
  && CUB_NS_QUALIFIER::transform::tile_eligible_v<
       Op, __detail::__unwrapped_value_t<OutIter>, sizeof...(InIters)>;

// Runtime predicate consulted by the cub::DeviceTransform tile hook before
// it commits to the tile path. Mirrors how CUB's dispatch_t::CanVectorize
// guards the vectorized kernel. The tile kernels use ct::assume_aligned<16>
// and ct::assume_divisible<16>, so violating these at runtime is UB.
// Returns false to tell the hook to fall back to the standard CUB dispatch.
template <typename OutIter, typename... InIters, typename OffsetT>
CUB_RUNTIME_FUNCTION bool
runtime_preconditions_ok(::cuda::std::tuple<InIters...> const& inputs, OutIter output, OffsetT num_items)
{
  // Pointer alignment is in bytes (for LDG.E.128); the kernel's
  // ct::assume_divisible<N> applies to num_items as an element count. These
  // are both 16 today by coincidence but live on different axes.
  constexpr int byte_align    = 16;
  constexpr int items_divisor = 16;

  auto out_ptr = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(output);
  const bool aligned_out = ::cuda::std::is_sufficiently_aligned<byte_align>(out_ptr);
  const bool aligned_in  = ::cuda::std::apply(
    [](auto... iters) {
      return ((::cuda::std::is_sufficiently_aligned<byte_align>(
                THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(iters)))
              && ...);
    },
    inputs);

  return aligned_out && aligned_in && (num_items % items_divisor) == 0;
}

// Bridge between cub::DeviceTransform::__transform_internal and the tile
// DeviceTransform above. Precondition: tile_dispatch_eligible_v<Op, OutIter,
// InIters...> is true AND runtime_preconditions_ok returned true. The kernel
// itself assumes 16-byte pointer alignment and num_items divisibility; the
// caller (the hook in device_transform.cuh) is responsible for checking
// runtime_preconditions_ok first.
//
// The tile kernel is launched with the trait's tile_op_type (a tile-friendly
// mirror of Op with __tile__ operator), NOT the user's Op instance -- the
// user's scalar functor cannot be invoked on ct::tile arguments.
template <typename TransformOp, typename OutIter, typename... InIters, typename OffsetT>
CUB_RUNTIME_FUNCTION cudaError_t dispatch(
  ::cuda::std::tuple<InIters...> inputs, OutIter output, OffsetT num_items, cudaStream_t stream)
{
  auto out_ptr = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(output);
  auto in_ptrs = ::cuda::std::apply(
    [](auto... iters) {
      return ::cuda::std::make_tuple(THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(iters)...);
    },
    inputs);
  using out_value_t = ::cuda::std::remove_cv_t<::cuda::std::remove_pointer_t<decltype(out_ptr)>>;
  using tile_op_t   =
    typename CUB_NS_QUALIFIER::transform::tile_eligible<TransformOp, out_value_t, sizeof...(InIters)>::tile_op_type;
  static_assert(::cuda::std::is_empty_v<tile_op_t>,
                "tile_op_type must be stateless (the tile kernel default-constructs it)");
  static_assert(::cuda::std::is_trivially_default_constructible_v<tile_op_t>,
                "tile_op_type must be trivially default constructible");

  return DeviceTransform::template Transform<0, CUB_NS_QUALIFIER::transform::tile_mufu_heavy_v<TransformOp>, tile_op_t>(
    in_ptrs, out_ptr, static_cast<::cuda::std::int64_t>(num_items), tile_op_t{}, stream);
}

} // namespace detail::transform::tile

CUB_NAMESPACE_END

#endif // _CCCL_CUB_HAS_TILE_TRANSFORM()
