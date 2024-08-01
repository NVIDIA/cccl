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

#include <cuda/std/span>
#include <cuda/std/tuple>

#include <cuda/experimental/hierarchy.cuh>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

template <typename Dimensions, typename... Options>
struct kernel_config;

namespace detail
{
struct launch_option
{
  static constexpr bool needs_attribute_space = false;
  static constexpr bool is_relevant_on_device = false;

protected:
  _CCCL_NODISCARD cudaError_t apply(cudaLaunchConfig_t&, void*) const noexcept
  {
    return cudaSuccess;
  }
};

template <typename Dimensions, typename... Options>
cudaError_t apply_kernel_config(
  const kernel_config<Dimensions, Options...>& config, cudaLaunchConfig_t& cuda_config, void* kernel) noexcept;

// Might need to go to the main namespace?
enum class launch_option_kind
{
  cooperative_launch,
  dynamic_shared_memory,
  launch_priority
};

struct option_not_found
{};

template <detail::launch_option_kind Kind>
struct find_option_in_tuple_impl
{
  template <typename Option, typename... Options>
  _CCCL_DEVICE auto& operator()(const Option& opt, const Options&... rest)
  {
    if constexpr (Option::kind == Kind)
    {
      return opt;
    }
    else
    {
      return (*this)(rest...);
    }
  }

  _CCCL_DEVICE auto operator()()
  {
    return option_not_found();
  }
};

template <detail::launch_option_kind Kind, typename... Options>
_CCCL_DEVICE auto& find_option_in_tuple(const ::cuda::std::tuple<Options...>& tuple)
{
  return ::cuda::std::apply(find_option_in_tuple_impl<Kind>(), tuple);
}

template <typename...>
inline constexpr bool no_duplicate_options = true;

template <typename Option, typename... Rest>
inline constexpr bool no_duplicate_options<Option, Rest...> =
  ((Option::kind != Rest::kind) && ...) && no_duplicate_options<Rest...>;

} // namespace detail

/**
 * @brief Launch option enabling cooperative launch
 *
 * This launch option causes the launched grid to be restricted to a number of
 * blocks that can simultaneously execute on the device. It means that every thread
 * in the launched grid can eventualy observe execution of each other thread in the grid.
 * It also enables usage of cooperative_groups::grid_group::sync() function, that
 * synchronizes all threads in the grid.
 *
 * @par Snippet
 * @code
 * #include <cudax/launch.cuh>
 * #include <cooperative_groups.h>
 *
 * template <typename Configuration>
 * __global__ void kernel(Configuration conf)
 * {
 *     auto grid = cooperative_groups::this_grid();
 *     grid.sync();
 * }
 *
 * void kernel_launch(cuda::stream_ref stream) {
 *     auto dims = cudax::make_hierarchy(cudax::block<128>(), cudax::grid(4));
 *     auto conf = cudax::make_configuration(dims, cooperative_launch());
 *
 *     cudax::launch(stream, conf, kernel);
 * }
 * @endcode
 */
struct cooperative_launch : public detail::launch_option
{
  static constexpr bool needs_attribute_space      = true;
  static constexpr bool is_relevant_on_device      = true;
  static constexpr detail::launch_option_kind kind = detail::launch_option_kind::cooperative_launch;

  constexpr cooperative_launch() = default;

  template <typename Dimensions, typename... Options>
  friend cudaError_t detail::apply_kernel_config(
    const kernel_config<Dimensions, Options...>& config, cudaLaunchConfig_t& cuda_config, void* kernel) noexcept;

private:
  _CCCL_NODISCARD cudaError_t apply(cudaLaunchConfig_t& config, void*) const noexcept
  {
    cudaLaunchAttribute attr;
    attr.id              = cudaLaunchAttributeCooperative;
    attr.val.cooperative = true;

    config.attrs[config.numAttrs++] = attr;

    return cudaSuccess;
  }
};

/**
 * @brief Launch option specyfying dynamic shared memory configuration
 *
 * This launch option causes the launch to allocate amount of shared memory sufficient
 * to store the specified number of object of the specified type.
 * This type can be constructed directly or with dynamic_shared_memory helper function.
 *
 * When launch configuration contains this option, that configuration can be then
 * passed to dynamic_smem_span or dynamic_smem_ref function to get a span/reference
 * to that shared memory allocation that is approprietly typed.
 * It is also possible to obtain that memory through the original
 * extern __shared__ variable[] declaration.
 *
 * CUDA guarantees that each device has at least 48kB of shared memory
 * per block, but most devices have more than that.
 * In order to allocate more dynamic shared memory than the portable
 * limit, opt-in NonPortableSize template argument should be set to true,
 * otherwise kernel launch will fail.
 *
 * @par Snippet
 * @code
 * #include <cudax/launch.cuh>
 *
 * template <typename Configuration>
 * __global__ void kernel(Configuration conf)
 * {
 *     auto dynamic_shared = cudax::dynamic_smem_span(conf);
 *     dynamic_shared[0] = 1;
 * }
 *
 * void kernel_launch(cuda::stream_ref stream) {
 *     auto dims = cudax::make_hierarchy(cudax::block<128>(), cudax::grid(4));
 *     auto conf = cudax::make_configuration(dims, dynamic_shared_memory<int, 128>());
 *
 *     cudax::launch(stream, conf, kernel);
 * }
 * @endcode
 * @par
 *
 * @tparam Content
 *  Type intended to be stored in dynamic shared memory
 *
 * @tparam Extent
 *  Statically specified number of Content objects in dynamic shared memory,
 *  or cuda::std::dynamic_extent, if its dynamic
 *
 * @tparam NonPortableSize
 *  Needs to be enabled to exceed the portable limit of 48kB of shared memory per block
 */
template <typename Content, std::size_t Extent = 1, bool NonPortableSize = false>
struct dynamic_shared_memory_option : public detail::launch_option
{
  using content_type                               = Content;
  static constexpr std::size_t extent              = Extent;
  static constexpr bool is_relevant_on_device      = true;
  static constexpr detail::launch_option_kind kind = detail::launch_option_kind::dynamic_shared_memory;
  const std::size_t size;

  constexpr dynamic_shared_memory_option(std::size_t set_size) noexcept
      : size(set_size)
  {}

  template <typename Dimensions, typename... Options>
  friend cudaError_t detail::apply_kernel_config(
    const kernel_config<Dimensions, Options...>& config, cudaLaunchConfig_t& cuda_config, void* kernel) noexcept;

private:
  _CCCL_NODISCARD cudaError_t apply(cudaLaunchConfig_t& config, void* kernel) const noexcept
  {
    cudaFuncAttributes attrs;
    int size_needed    = static_cast<int>(size * sizeof(Content));
    cudaError_t status = cudaFuncGetAttributes(&attrs, kernel);

    if ((size_needed > attrs.maxDynamicSharedSizeBytes) && NonPortableSize)
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

/**
 * @brief Creates an instance of dynamic_shared_memory_option with a statically known size
 *
 * Type and size need to specified using template arguments.
 *
 * @tparam Content
 *  Type intended to be stored in dynamic shared memory
 *
 * @tparam Extent
 *  Statically specified number of Content objects in dynamic shared memory
 *
 * @tparam NonPortableSize
 *  Needs to be enabled to exceed the portable limit of 48kB of shared memory per block
 */
template <typename Content, std::size_t Extent = 1, bool NonPortableSize = false>
constexpr dynamic_shared_memory_option<Content, Extent, NonPortableSize> dynamic_shared_memory() noexcept
{
  static_assert(Extent != ::cuda::std::dynamic_extent, "Size needs to be provided when dynamic_extent is specified");

  return dynamic_shared_memory_option<Content, Extent, NonPortableSize>(Extent);
}

/**
 * @brief Creates an instance of dynamic_shared_memory_option with a dynamic size
 *
 * Type stored needs to be specified using template argument, while size is a function argument
 *
 * @param count
 *  Number of Content elements in dynamic shared memory
 *
 * @tparam Content
 *  Type intended to be stored in dynamic shared memory
 *
 * @tparam NonPortableSize
 *  Needs to be enabled to exceed the portable limit of 48kB of shared memory per block
 */
template <typename Content, bool NonPortableSize = false>
constexpr dynamic_shared_memory_option<Content, ::cuda::std::dynamic_extent, NonPortableSize>
dynamic_shared_memory(std::size_t count) noexcept
{
  return dynamic_shared_memory_option<Content, ::cuda::std::dynamic_extent, NonPortableSize>(count);
}

/**
 * @brief Launch option specifying launch priority
 *
 * This launch option causes the launched grid to be scheduled with the specified priority.
 * More about stream priorities and valid values can be found in the CUDA programming guide
 * `here <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-priorities>`_
 */
struct launch_priority : public detail::launch_option
{
  static constexpr bool needs_attribute_space      = true;
  static constexpr bool is_relevant_on_dpevice     = false;
  static constexpr detail::launch_option_kind kind = detail::launch_option_kind::launch_priority;
  unsigned int priority;

  launch_priority(unsigned int p) noexcept
      : priority(p)
  {}

  template <typename Dimensions, typename... Options>
  friend cudaError_t detail::apply_kernel_config(
    const kernel_config<Dimensions, Options...>& config, cudaLaunchConfig_t& cuda_config, void* kernel) noexcept;

private:
  _CCCL_NODISCARD cudaError_t apply(cudaLaunchConfig_t& config, void*) const noexcept
  {
    cudaLaunchAttribute attr;
    attr.id           = cudaLaunchAttributePriority;
    attr.val.priority = priority;

    config.attrs[config.numAttrs++] = attr;

    return cudaSuccess;
  }
};

/**
 * @brief Type describing a kernel launch configuration
 *
 * This type should not be constructed directly and make_config helper function should be used instead
 *
 * @tparam Dimensions
 * cuda::experimetnal::hierarchy_dimensions instance that describes dimensions of thread hierarchy in this configuration
 * object
 *
 * @tparam Options
 * Types of options that were added to this configuration object
 */
template <typename Dimensions, typename... Options>
struct kernel_config
{
  Dimensions dims;
  ::cuda::std::tuple<Options...> options;

  static_assert(::cuda::std::_Or<std::true_type, ::cuda::std::is_base_of<detail::launch_option, Options>...>::value);
  static_assert(detail::no_duplicate_options<Options...>);

  constexpr kernel_config(const Dimensions& dims, const Options&... opts)
      : dims(dims)
      , options(opts...) {};
  constexpr kernel_config(const Dimensions& dims, const ::cuda::std::tuple<Options...>& opts)
      : dims(dims)
      , options(opts) {};

  /**
   * @brief Add a new option to this configuration
   *
   * Returns a new kernel_config that has all option and dimensions from this kernel_config
   * with the option from the argument added to it
   *
   * @param new_option
   * Option to be added to the configuration
   */
  template <typename... NewOptions>
  _CCCL_NODISCARD auto add(const NewOptions&... new_options) const
  {
    return kernel_config<Dimensions, Options..., NewOptions...>(
      dims, ::cuda::std::tuple_cat(options, ::cuda::std::make_tuple(new_options...)));
  }
};

template <typename Dimensions,
          typename... Options,
          typename Option,
          typename = ::cuda::std::enable_if_t<::cuda::std::is_base_of_v<detail::launch_option, Option>>>
_CCCL_NODISCARD constexpr auto
operator&(const kernel_config<Dimensions, Options...>& config, const Option& option) noexcept
{
  return config.add(option);
}

template <typename... Levels,
          typename Option,
          typename = ::cuda::std::enable_if_t<::cuda::std::is_base_of_v<detail::launch_option, Option>>>
_CCCL_NODISCARD constexpr auto operator&(const hierarchy_dimensions<Levels...>& dims, const Option& option) noexcept
{
  return kernel_config(dims, option);
}

/**
 * @brief Construct kernel configuration
 *
 * This function takes thread hierarchy dimensions description and any number of launch options and combines
 * them into kernel configuration object. It can be then used along with kernel function and its argument to launch
 * that kernel with the specified dimensions and options
 *
 * @param dims
 * Object describing dimensions of the thread hierarchy in the resulting kernel configuration object
 *
 * @param opts
 * Variadic number of launch configuration options to be included in the resulting kernel configuration object
 */
template <typename... Levels, typename... Opts>
_CCCL_NODISCARD constexpr auto make_config(const hierarchy_dimensions<Levels...>& dims, const Opts&... opts) noexcept
{
  return kernel_config<hierarchy_dimensions<Levels...>, Opts...>(dims, opts...);
}

namespace detail
{

template <typename Dimensions, typename... Options>
inline unsigned int constexpr kernel_config_count_attr_space(const kernel_config<Dimensions, Options...>&) noexcept
{
  return (0 + ... + Options::needs_attribute_space);
}

template <typename Dimensions, typename... Options>
_CCCL_NODISCARD cudaError_t apply_kernel_config(
  const kernel_config<Dimensions, Options...>& config, cudaLaunchConfig_t& cuda_config, void* kernel) noexcept
{
  cudaError_t status = cudaSuccess;

  ::cuda::std::apply(
    [&](auto&... config_options) {
      // Use short-cutting && to skip the rest on error, is this too convoluted?
      (void) (... && [&](cudaError_t call_status) {
        status = call_status;
        return call_status == cudaSuccess;
      }(config_options.apply(cuda_config, kernel)));
    },
    config.options);

  return status;
}

// Needs to be a char casted to the apropriate type, if it would be a template
//  different instantiations would clash the extern symbol
_CCCL_DEVICE _CCCL_NODISCARD static char* get_smem_ptr() noexcept
{
  extern __shared__ char dynamic_smem[];

  return &dynamic_smem[0];
}
} // namespace detail

// Might consider cutting this one due to being a potential trap with missing & in auto& var = dynamic_smem_ref(...);
/**
 * @brief Returns a reference to shared memory variable in dynamic shared memory
 *
 * This function returns a reference to a variable placed in dynamic shared memory.
 * It accepts a kernel_config containing a dynamic_shared_memory_option.
 * Its only usable when dynamic shared memory option is holding a single object.
 */
template <typename Dimensions, typename... Options>
_CCCL_DEVICE auto& dynamic_smem_ref(const kernel_config<Dimensions, Options...>& config) noexcept
{
  auto& option      = detail::find_option_in_tuple<detail::launch_option_kind::dynamic_shared_memory>(config.options);
  using option_type = ::cuda::std::remove_reference_t<decltype(option)>;
  static_assert(!::cuda::std::is_same_v<option_type, detail::option_not_found>,
                "Dynamic shared memory option not found in the kernel configuration");
  static_assert(option_type::extent == 1, "Usable only on dynamic shared memory with a single element");

  return *reinterpret_cast<typename option_type::content_type*>(detail::get_smem_ptr());
}

/**
 * @brief Returns a cuda::std::span object refering to dynamic shared memory region
 *
 * This function returns a std::std::span object refering to the dynamic shared memory region
 * configured when launching the kernel.
 * It accepts a kernel_config containing a dynamic_shared_memory_option.
 * It is typed and sized according to the launch option provided as input.
 */
template <typename Dimensions, typename... Options>
_CCCL_DEVICE auto dynamic_smem_span(const kernel_config<Dimensions, Options...>& config) noexcept
{
  auto& option      = detail::find_option_in_tuple<detail::launch_option_kind::dynamic_shared_memory>(config.options);
  using option_type = ::cuda::std::remove_reference_t<decltype(option)>;
  static_assert(!::cuda::std::is_same_v<option_type, detail::option_not_found>,
                "Dynamic shared memory option not found in the kernel configuration");

  return cuda::std::span<typename option_type::content_type, option_type::extent>(
    reinterpret_cast<typename option_type::content_type*>(detail::get_smem_ptr()), option.size);
}

} // namespace cuda::experimental
#endif // _CCCL_STD_VER >= 2017
#endif // _CUDAX__LAUNCH_CONFIGURATION
