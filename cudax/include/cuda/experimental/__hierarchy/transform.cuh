//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__HIERARCHY_TRANSFORM
#define _CUDAX__HIERARCHY_TRANSFORM

#include <cuda/experimental/__hierarchy/hierarchy_dimensions.cuh>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

// Kind of a hack to fit with rest of the code (for now? ;)
template <typename Unit>
struct at_least : dimensions<unsigned int, ::cuda::std::dynamic_extent, 1, 1>
{
  using unit = Unit;
  _CCCL_HOST_DEVICE constexpr at_least(size_t count, const Unit& level = Unit())
      : dimensions<unsigned int, ::cuda::std::dynamic_extent, 1, 1>(count)
  {}
};

namespace detail
{

template <typename Level>
struct dimensions_handler<at_least<Level>>
{
  static constexpr bool is_type_supported = true;

  _CCCL_NODISCARD _CCCL_HOST_DEVICE static constexpr auto translate(const at_least<Level>& d) noexcept
  {
    return d;
  }
};

template <typename T>
inline constexpr bool is_metadims_at_least = false;

template <typename Unit>
inline constexpr bool is_metadims_at_least<at_least<Unit>> = true;

template <typename What, typename ByWhat>
auto ceil_div(What what, ByWhat by_what)
{
  return (what + by_what - 1) / by_what;
}

/*
template <typename Level, typename RestTransformed>
_CCCL_NODISCARD constexpr auto level_transform(const Level& level, const RestTransformed& rest)
{}
*/
_CCCL_NODISCARD constexpr auto hierarchy_transform_impl()
{
  return ::cuda::std::make_tuple();
}

template <typename T>
struct check;

template <typename L1, typename... Rest>
_CCCL_NODISCARD constexpr auto hierarchy_transform_impl(const L1& level, const Rest&... rest)
{
  auto rest_transformed = hierarchy_transform_impl(rest...);

  using dims_type = ::cuda::std::decay_t<decltype(level.dims)>;

  if constexpr (detail::is_metadims_at_least<dims_type>)
  {
    auto tmp_hierarchy         = hierarchy_dimensions(rest_transformed);
    auto count_of_below_levels = tmp_hierarchy.template count<typename dims_type::unit>();
    auto new_dims              = dimensions<dimensions_index_type, ::cuda::std::dynamic_extent, 1, 1>(
      ceil_div(level.dims.extent(0), count_of_below_levels));

    return ::cuda::std::tuple_cat(::cuda::std::make_tuple(level.transform(new_dims)), rest_transformed);
  }
  else
  {
    static_assert(detail::usable_for_queries<dims_type>);
    return ::cuda::std::tuple_cat(::cuda::std::make_tuple(level), rest_transformed);
  }
}
} // namespace detail

template <typename... Levels>
_CCCL_NODISCARD constexpr auto hierarchy_transform(
  const hierarchy_dimensions<Levels...>& hierarchy /* will also take a kernel function and arguments */)
{
  return hierarchy_dimensions(::cuda::std::apply(detail::hierarchy_transform_impl<Levels...>, hierarchy.levels));
}

} // namespace cuda::experimental
#endif
#endif
