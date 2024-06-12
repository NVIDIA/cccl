//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAUNCH_CONFIGURATION
#define _CUDAX__LAUNCH_CONFIGURATION
#include <cuda/experimental/hierarchy.cuh>
#include <cuda/std/span>
#include <cuda/std/tuple>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

namespace detail
{
struct launch_option
{
  static constexpr bool needs_attribute_space = false;
  static constexpr bool is_relevant_on_device = false;

  _CCCL_NODISCARD cudaError_t apply(cudaLaunchConfig_t& config, void* kernel) const noexcept
  {
    return cudaSuccess;
  }
};
} // namespace detail

struct cooperative_launch_option : public detail::launch_option
{
  static constexpr bool needs_attribute_space = true;
  static constexpr bool is_relevant_on_device = true;

  constexpr cooperative_launch_option() = default;

  _CCCL_NODISCARD cudaError_t apply(cudaLaunchConfig_t& config, void* kernel) const noexcept
  {
    cudaLaunchAttribute attr;
    attr.id              = cudaLaunchAttributeCooperative;
    attr.val.cooperative = true;

    config.attrs[config.numAttrs++] = attr;

    return cudaSuccess;
  }
};

constexpr cooperative_launch_option cooperative_launch() noexcept
{
  return cooperative_launch_option();
}

template <typename Content, std::size_t Extent = 1>
struct dyn_smem_option : public detail::launch_option
{
  const std::size_t size                      = Extent;
  static constexpr bool is_relevant_on_device = true;

  constexpr dyn_smem_option(std::size_t set_size) noexcept
      : size(set_size)
  {}

  _CCCL_NODISCARD cudaError_t apply(cudaLaunchConfig_t& config, void* kernel) const noexcept
  {
    cudaFuncAttributes attrs;
    std::size_t size_needed = size * sizeof(Content);
    cudaError_t status      = cudaFuncGetAttributes(&attrs, kernel);

    if (size_needed > static_cast<std::size_t>(attrs.maxDynamicSharedSizeBytes))
    {
      // TODO since 12.6 there is a per launch option available, we should switch once compatibility is not an issue
      // TODO should we validate the max amount with device props or just pass it through and rely on driver error?
      status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, size_needed);
      if (status != cudaSuccess)
      {
        return status;
      }
    }

    config.dynamicSmemBytes = size_needed;
    return cudaSuccess;
  }
};

// Functions to create dyn_smem_option, should all config elements be created with a function?
template <typename Content, std::size_t Extent = 1>
constexpr dyn_smem_option<Content, Extent> dynamic_shared_memory() noexcept
{
  static_assert(Extent != cuda::std::dynamic_extent, "Size needs to be provided when dynamic_extent is specified");

  return dyn_smem_option<Content, Extent>(Extent);
}

template <typename Content>
constexpr dyn_smem_option<Content, cuda::std::dynamic_extent> dynamic_shared_memory(std::size_t size) noexcept
{
  return dyn_smem_option<Content, cuda::std::dynamic_extent>(size);
}

struct launch_priority_config_element : public detail::launch_option
{
  static constexpr bool needs_attribute_space = true;
  static constexpr bool is_relevant_on_device = false;
  unsigned int priority;

  launch_priority_config_element(unsigned int p) noexcept
      : priority(p)
  {}

  _CCCL_NODISCARD cudaError_t apply(cudaLaunchConfig_t& config, void* kernel) const noexcept
  {
    cudaLaunchAttribute attr;
    attr.id           = cudaLaunchAttributePriority;
    attr.val.priority = priority;

    config.attrs[config.numAttrs++] = attr;

    return cudaSuccess;
  }
};

static launch_priority_config_element launch_priority(unsigned int p) noexcept
{
  return launch_priority_config_element(p);
}

template <typename Dimensions, typename... Options>
struct kernel_config : public Options...
{
  Dimensions dims;

  static_assert(cuda::std::_Or<std::true_type, cuda::std::is_base_of<detail::launch_option, Options>...>::value);

  constexpr kernel_config(const Dimensions& dims, const Options&... opts) noexcept
      : Options(opts)...
      , dims(dims){};

  static unsigned int constexpr num_attrs_needed() noexcept
  {
    return (0 + ... + Options::needs_attribute_space);
  }

  template <typename Option>
  _CCCL_NODISCARD auto add(const Option& new_option)
  {
    return kernel_config(dims, static_cast<Options>(*this)..., new_option);
  }

  _CCCL_NODISCARD cudaError_t apply(cudaLaunchConfig_t& config, void* kernel) const noexcept
  {
    cudaError_t status = cudaSuccess;

    // Use short-cutting && to skip the rest on error, is this too convoluted?
    (void) (... && [&](cudaError_t call_status) {
      status = call_status;
      return call_status == cudaSuccess;
    }(Options::apply(config, kernel)));

    return status;
  }
};

template <typename Dimensions,
          typename... Options,
          typename Option,
          typename = cuda::std::enable_if_t<cuda::std::is_base_of_v<detail::launch_option, Option>>>
_CCCL_NODISCARD constexpr auto
operator&(const kernel_config<Dimensions, Options...>& config, const Option& option) noexcept
{
  return config.add(option);
}

template <typename... Levels,
          typename Option,
          typename = cuda::std::enable_if_t<cuda::std::is_base_of_v<detail::launch_option, Option>>>
_CCCL_NODISCARD constexpr auto operator&(const hierarchy_dimensions<Levels...>& dims, const Option& option) noexcept
{
  return kernel_config(dims, option);
}

template <typename... Levels, typename... Opts>
_CCCL_NODISCARD constexpr auto make_config(const hierarchy_dimensions<Levels...>& dims, const Opts&... opts) noexcept
{
  return kernel_config<hierarchy_dimensions<Levels...>, Opts...>(dims, opts...);
}

_CCCL_DEVICE _CCCL_NODISCARD static char* get_smem_ptr() noexcept
{
  extern __shared__ char dynamic_smem[];

  return &dynamic_smem[0];
}

// Might consider cutting this one due to being a potential trap with missing & in auto& var = dynamic_smem_ref(...);
template <typename Content>
_CCCL_DEVICE _CCCL_NODISCARD Content& dynamic_smem_ref(const dyn_smem_option<Content, 1>& m) noexcept
{
  return *reinterpret_cast<Content*>(get_smem_ptr());
}

template <typename Content, std::size_t Extent>
_CCCL_DEVICE _CCCL_NODISCARD ::cuda::std::span<Content, Extent>
dynamic_smem_span(const dyn_smem_option<Content, Extent>& m) noexcept
{
  return cuda::std::span<Content, Extent>(reinterpret_cast<Content*>(get_smem_ptr()), m.size);
}

} // namespace cuda::experimental
#endif // _CCCL_STD_VER >= 2017
#endif // _CUDAX__LAUNCH_CONFIGURATION
