// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal dispatch helpers for cub::DeviceTransform's tile path:
//   tile_dispatch_eligible_v  -- compile-time predicate the hook consults
//   runtime_preconditions_valid  -- runtime alignment + divisibility predicate
//   dispatch                  -- bridge that picks the tile size and launches
//                                the tile kernel with the trait's substitute functor
// Extension points (tile_eligible / tile_operator / tile_mufu_heavy) live in
// dispatch_transform_tile_traits.cuh under cub::detail::transform::tile.
// Requires CTK 13.4 or newer and nvcc invoked with --enable-tile.

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
#  include <cub/util_debug.cuh>
#  include <cub/util_device.cuh>

#  include <thrust/type_traits/is_contiguous_iterator.h>
#  include <thrust/type_traits/unwrap_contiguous_iterator.h>

#  include <cuda/__cmath/ceil_div.h>
#  include <cuda/__device/compute_capability.h>
#  include <cuda/std/__iterator/readable_traits.h>
#  include <cuda/std/__memory/is_sufficiently_aligned.h>
#  include <cuda/std/__tuple_dir/apply.h>
#  include <cuda/std/__type_traits/is_empty.h>
#  include <cuda/std/__type_traits/is_trivially_default_constructible.h>
#  include <cuda/std/__utility/integer_sequence.h>
#  include <cuda/std/cstdint>
#  include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

namespace detail::transform::tile
{
template <int TileSize, typename Fn, typename Out, typename... Ins, ::cuda::std::size_t... Idx>
[[nodiscard]] ::cudaError_t launch_impl(
  ::cuda::std::tuple<Ins*...> inputs,
  Out* output,
  ::cuda::std::int64_t num_items,
  ::cudaStream_t stream,
  ::cuda::std::index_sequence<Idx...>)
{
  if (num_items <= 0)
  {
    return ::cudaSuccess;
  }

  // One CTA per tile. The cast to the unsigned grid x-dim can't truncate: num_blocks > 2^32-1
  // would need num_items > TileSize * 2^32 (>= 2^40 elements), more than any device can hold.
  const ::cuda::std::int64_t num_blocks = ::cuda::ceil_div(num_items, ::cuda::std::int64_t{TileSize});

  cub::detail::transform::tile::transform_kernel<TileSize, Fn>
    <<<static_cast<unsigned>(num_blocks), 1, 0, stream>>>(num_items, output, ::cuda::std::get<Idx>(inputs)...);

  return CubDebug(::cudaGetLastError());
}

// Combined compile-time predicate for whether (Op, OutIter, InIters...) can use the tile path. We use this with
// `if constexpr` for dispatch: when true the hook tries the tile kernel first and, on runtime alignment/divisibility
// failure, falls through to the standard CUB dispatch; when false the tile branch is discarded entirely.
template <typename Op, typename OutIter, typename... InIters>
inline constexpr bool tile_dispatch_eligible_v =
  THRUST_NS_QUALIFIER::is_contiguous_iterator_v<OutIter>
  && (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<InIters> && ...)
  && ::cuda::std::is_empty_v<Op> && tile_eligible_v<Op, ::cuda::std::iter_value_t<OutIter>, sizeof...(InIters)>;

// Runtime arch gate: tile needs sm_80+. False (fall back to CUB) below sm_80 or if the cc query fails.
[[nodiscard]] CUB_RUNTIME_FUNCTION inline bool device_supports_tile()
{
  ::cuda::compute_capability cc{};
  return cub::detail::ptx_compute_cap(cc) == ::cudaSuccess && cc >= ::cuda::compute_capability{8, 0};
}

// Runtime precondition the tile hook checks before dispatching: 16-byte pointer alignment + num_items % 16 == 0
// (the kernels assume_aligned<16>/assume_divisible<16>, so violating these is UB). False -> fall back to CUB.
template <typename OutIter, typename... InIters, typename OffsetT>
[[nodiscard]] CUB_RUNTIME_FUNCTION bool
runtime_preconditions_valid(::cuda::std::tuple<InIters...> const& inputs, OutIter output, OffsetT num_items)
{
  // Pointer alignment is in bytes (for LDG.E.128); the kernel's
  // ct::assume_divisible<N> applies to num_items as an element count. These
  // are both 16 today by coincidence but live on different axes.
  constexpr int byte_align    = 16;
  constexpr int items_divisor = 16;

  const auto out_ptr     = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(output);
  const bool aligned_out = ::cuda::std::is_sufficiently_aligned<byte_align>(out_ptr);
  const bool aligned_in  = ::cuda::std::apply(
    [](auto... iters) {
      return (
        (::cuda::std::is_sufficiently_aligned<byte_align>(THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(iters)))
        && ...);
    },
    inputs);

  return aligned_out && aligned_in && (num_items % items_divisor) == 0;
}

// Bridge from cub::DeviceTransform::__transform_internal to the tile kernel. Precondition (the caller
// checks it): tile_dispatch_eligible_v is true AND runtime_preconditions_valid returned true. Launches the kernel
// with tile_operator_t<Op> -- Op's registered __tile__ mirror (a scalar functor can't be invoked on ct::tile).
template <typename TransformOp, typename OutIter, typename... InIters, typename OffsetT>
[[nodiscard]] CUB_RUNTIME_FUNCTION ::cudaError_t
dispatch(::cuda::std::tuple<InIters...> inputs, OutIter output, OffsetT num_items, ::cudaStream_t stream)
{
  if (num_items <= 0)
  {
    return ::cudaSuccess;
  }

  const auto out_ptr = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(output);
  const auto in_ptrs = ::cuda::std::apply(
    [](auto... iters) {
      return ::cuda::std::make_tuple(THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(iters)...);
    },
    inputs);

  using tile_op_t = tile_operator_t<TransformOp>;
  static_assert(::cuda::std::is_empty_v<tile_op_t>,
                "tile_operator type must be stateless (the tile kernel default-constructs it)");
  static_assert(::cuda::std::is_trivially_default_constructible_v<tile_op_t>,
                "tile_operator type must be trivially default constructible");

  constexpr int tile_size =
    cub::detail::transform::tile::pick_tile_size<::cuda::std::iter_value_t<OutIter>,
                                                 ::cuda::std::iter_value_t<InIters>...>(tile_mufu_heavy_v<TransformOp>);
  return cub::detail::transform::tile::launch_impl<tile_size, tile_op_t>(
    in_ptrs,
    out_ptr,
    static_cast<::cuda::std::int64_t>(num_items),
    stream,
    ::cuda::std::index_sequence_for<InIters...>{});
}
} // namespace detail::transform::tile

CUB_NAMESPACE_END

#endif // _CCCL_CUB_HAS_TILE_TRANSFORM()
