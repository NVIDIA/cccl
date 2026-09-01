// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/hierarchy>

#include <vector>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

using scalar_hierarchy = decltype(cuda::make_hierarchy(cuda::grid_dims(1U), cuda::block_dims(1U)));

struct empty_mixin
{};

// Model the documented extension point for descriptors with library-specific
// metadata. The empty mixin also shares offset zero with the descriptor base.
struct derived_block_desc
    : empty_mixin
    , cuda::hierarchy_level_desc<cuda::block_level, cuda::std::extents<cuda::dimensions_index_type, 91, 92, 93>>
{
  int metadata = 7;
};

[[gnu::noinline]] void inspect_static(const decltype(cuda::make_hierarchy(
  cuda::grid_dims<2, 3, 4>(), cuda::cluster_dims<5, 6, 7>(), cuda::block_dims<8, 9, 10>()))& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_dynamic(const decltype(cuda::make_hierarchy(
  cuda::grid_dims(dim3{1, 1, 1}), cuda::cluster_dims(dim3{1, 1, 1}), cuda::block_dims(dim3{1, 1, 1})))& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void
inspect_partial(const decltype(cuda::make_hierarchy<cuda::block_level>(cuda::grid_dims<14, 15, 16>()))& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void
inspect_derived(const decltype(cuda::make_hierarchy(cuda::grid_dims<81, 82, 83>(), derived_block_desc{}))& values)
{
  KEEP_FOR_DEBUGGER(values);
}

// An optimized build inlines every use of std::vector::operator[], so no
// out-of-line copy is left for the debugger to call. Instantiate that member
// explicitly to keep a copy, which lets the debugger evaluate values[i].
template
  typename std::vector<scalar_hierarchy>::const_reference std::vector<scalar_hierarchy>::operator[](size_type) const;

[[gnu::noinline]] void inspect_vector(const std::vector<scalar_hierarchy>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_before_update(const scalar_hierarchy& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_after_update(const scalar_hierarchy& values)
{
  KEEP_FOR_DEBUGGER(values);
}

int main()
{
  const auto static_values =
    cuda::make_hierarchy(cuda::grid_dims<2, 3, 4>(), cuda::cluster_dims<5, 6, 7>(), cuda::block_dims<8, 9, 10>());
  inspect_static(static_values);

  const auto dynamic_values = cuda::make_hierarchy(
    cuda::grid_dims(dim3{11, 12, 13}), cuda::cluster_dims(dim3{21, 22, 23}), cuda::block_dims(dim3{31, 32, 33}));
  inspect_dynamic(dynamic_values);

  const auto partial_values = cuda::make_hierarchy<cuda::block_level>(cuda::grid_dims<14, 15, 16>());
  inspect_partial(partial_values);

  const auto derived_values = cuda::make_hierarchy(cuda::grid_dims<81, 82, 83>(), derived_block_desc{});
  inspect_derived(derived_values);

  const std::vector<scalar_hierarchy> hierarchy_vector{
    cuda::make_hierarchy(cuda::grid_dims(41U), cuda::block_dims(42U)),
    cuda::make_hierarchy(cuda::grid_dims(51U), cuda::block_dims(52U))};
  inspect_vector(hierarchy_vector);

  auto mutable_values = cuda::make_hierarchy(cuda::grid_dims(61U), cuda::block_dims(62U));
  inspect_before_update(mutable_values);
  mutable_values = cuda::make_hierarchy(cuda::grid_dims(71U), cuda::block_dims(72U));
  inspect_after_update(mutable_values);
}
