//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAUNCH_CONFIGURATION
#define _CUDAX__LAUNCH_CONFIGURATION

#include <cuda/std/__execution/env.h>
#include <cuda/std/span>
#include <cuda/std/tuple>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/hierarchy.cuh>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

template <typename _Dimensions, typename... _Options>
struct kernel_config;

template <typename _Option>
struct __launch_option_wrapper
{
  _Option __option_;

  constexpr __launch_option_wrapper(const _Option& __option) noexcept
      : __option_(__option)
  {}

  constexpr _Option _CCCL_HOST_DEVICE query(const _Option::__tag&) const noexcept
  {
    return __option_;
  }
};

namespace __detail
{
struct launch_option
{
  static constexpr bool needs_attribute_space = false;
  static constexpr bool is_relevant_on_device = false;

protected:
  [[nodiscard]] cudaError_t apply(cudaLaunchConfig_t&, void*) const noexcept
  {
    return cudaSuccess;
  }
};

template <typename _Dimensions, typename... _Options>
cudaError_t apply_kernel_config(
  const kernel_config<_Dimensions, _Options...>& __config, cudaLaunchConfig_t& __cuda_config, void* __kernel) noexcept;

// Might need to go to the main namespace?
enum class launch_option_kind
{
  cooperative_launch,
  dynamic_shared_memory,
  launch_priority
};

struct option_not_found
{};

template <typename _Option, typename... _OptionsList>
inline constexpr bool __option_present_in_list = ((_Option::kind == _OptionsList::kind) || ...);

template <typename...>
inline constexpr bool no_duplicate_options = true;

template <typename _Option, typename... _Rest>
inline constexpr bool no_duplicate_options<_Option, _Rest...> =
  !__option_present_in_list<_Option, _Rest...> && no_duplicate_options<_Rest...>;

} // namespace __detail

/**
 * @brief Launch option enabling cooperative launch
 *
 * This launch option causes the launched grid to be restricted to a number of
 * blocks that can simultaneously execute on the device. It means that every thread
 * in the launched grid can eventually observe execution of each other thread in the grid.
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
struct cooperative_launch_option : public __detail::launch_option
{
  static constexpr bool needs_attribute_space        = true;
  static constexpr bool is_relevant_on_device        = true;
  static constexpr __detail::launch_option_kind kind = __detail::launch_option_kind::cooperative_launch;
  using __tag                                        = cooperative_launch_option;

  constexpr cooperative_launch_option() = default;
  constexpr _CCCL_HOST_DEVICE cooperative_launch_option(const cooperative_launch_option&) noexcept {}

  constexpr cooperative_launch_option operator()() const noexcept
  {
    return cooperative_launch_option();
  }

  template <typename _Dimensions, typename... _Options>
  friend cudaError_t __detail::apply_kernel_config(
    const kernel_config<_Dimensions, _Options...>& __config, cudaLaunchConfig_t& __cuda_config, void* __kernel) noexcept;

private:
  [[nodiscard]] cudaError_t apply(cudaLaunchConfig_t& __config, void*) const noexcept
  {
    cudaLaunchAttribute __attr;
    __attr.id              = cudaLaunchAttributeCooperative;
    __attr.val.cooperative = true;

    __config.attrs[__config.numAttrs++] = __attr;

    return cudaSuccess;
  }
};

inline constexpr cooperative_launch_option cooperative_launch;

/**
 * @brief Launch option specifying dynamic shared memory configuration
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
template <typename _Content, std::size_t _Extent = 1, bool _NonPortableSize = false>
struct dynamic_shared_memory_option : public __detail::launch_option
{
  using content_type                                 = _Content;
  static constexpr std::size_t extent                = _Extent;
  static constexpr bool is_relevant_on_device        = true;
  static constexpr __detail::launch_option_kind kind = __detail::launch_option_kind::dynamic_shared_memory;
  const std::size_t size                             = _Extent == ::cuda::std::dynamic_extent ? 0 : _Extent;

  constexpr dynamic_shared_memory_option() = default;

  constexpr _CCCL_HOST_DEVICE dynamic_shared_memory_option(const dynamic_shared_memory_option& __other) noexcept
      : size(__other.size)
  {}

  constexpr dynamic_shared_memory_option(std::size_t __set_size) noexcept
      : size(__set_size)
  {}

  template <typename _Dimensions, typename... _Options>
  friend cudaError_t __detail::apply_kernel_config(
    const kernel_config<_Dimensions, _Options...>& __config, cudaLaunchConfig_t& __cuda_config, void* __kernel) noexcept;

private:
  [[nodiscard]] cudaError_t apply(cudaLaunchConfig_t& __config, void* __kernel) const noexcept
  {
    cudaFuncAttributes __attrs;
    int __size_needed    = static_cast<int>(size * sizeof(_Content));
    cudaError_t __status = cudaFuncGetAttributes(&__attrs, __kernel);

    if ((__size_needed > __attrs.maxDynamicSharedSizeBytes) && _NonPortableSize)
    {
      // TODO since 12.6 there is a per launch option available, we should switch once compatibility is not an issue
      // TODO should we validate the max amount with device props or just pass it through and rely on driver error?
      __status = cudaFuncSetAttribute(__kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __size_needed);
      if (__status != cudaSuccess)
      {
        return __status;
      }
    }

    __config.dynamicSmemBytes = __size_needed;
    return cudaSuccess;
  }
};

template <typename _Content, std::size_t _Extent, bool _NonPortableSize>
struct __dynamic_shared_memory_t
{
  constexpr dynamic_shared_memory_option<_Content, _Extent, _NonPortableSize> operator()() const noexcept
  {
    return dynamic_shared_memory_option<_Content, _Extent, _NonPortableSize>();
  }
};

template <>
struct __dynamic_shared_memory_t<void, ::cuda::std::dynamic_extent, false>
{};

template <typename _Content, bool _NonPortableSize>
struct __dynamic_shared_memory_t<_Content, ::cuda::std::dynamic_extent, _NonPortableSize>
{
  constexpr dynamic_shared_memory_option<_Content, ::cuda::std::dynamic_extent, _NonPortableSize>
  operator()(std::size_t __size) const noexcept
  {
    return dynamic_shared_memory_option<_Content, ::cuda::std::dynamic_extent, _NonPortableSize>(__size);
  }
};

template <typename _Content, std::size_t _Extent, bool _NonPortableSize>
struct __launch_option_wrapper<dynamic_shared_memory_option<_Content, _Extent, _NonPortableSize>>
{
  dynamic_shared_memory_option<_Content, _Extent, _NonPortableSize> __option_;

  constexpr __launch_option_wrapper(
    const dynamic_shared_memory_option<_Content, _Extent, _NonPortableSize>& __option) noexcept
      : __option_(__option)
  {}

  constexpr dynamic_shared_memory_option<_Content, _Extent, _NonPortableSize> _CCCL_HOST_DEVICE
  query(const __dynamic_shared_memory_t<_Content, _Extent, _NonPortableSize>&) const noexcept
  {
    return __option_;
  }

  constexpr dynamic_shared_memory_option<_Content, _Extent, _NonPortableSize> _CCCL_HOST_DEVICE
  query(const __dynamic_shared_memory_t<void, ::cuda::std::dynamic_extent, _NonPortableSize>&) const noexcept
  {
    return __option_;
  }
};

template <typename _Content = void, std::size_t _Extent = ::cuda::std::dynamic_extent, bool _NonPortableSize = false>
inline constexpr __dynamic_shared_memory_t<_Content, _Extent, _NonPortableSize> dynamic_shared_memory;

#  if _CCCL_DOXYGEN_INVOKED
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
template <typename _Content, std::size_t _Extent, bool _NonPortableSize = false>
constexpr dynamic_shared_memory_option<_Content, _Extent, _NonPortableSize> dynamic_shared_memory() noexcept
{
  return dynamic_shared_memory_option<_Content, _Extent, _NonPortableSize>(_Extent);
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
template <typename _Content, bool _NonPortableSize = false>
constexpr dynamic_shared_memory_option<_Content, ::cuda::std::dynamic_extent, _NonPortableSize>
dynamic_shared_memory(std::size_t __count) noexcept
{
  return dynamic_shared_memory_option<_Content, ::cuda::std::dynamic_extent, _NonPortableSize>(__count);
}
#  endif

struct __launch_priority_t;

/**
 * @brief Launch option specifying launch priority
 *
 * This launch option causes the launched grid to be scheduled with the specified priority.
 * More about stream priorities and valid values can be found in the CUDA programming guide
 * `here <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-priorities>`_
 */
struct launch_priority_option : public __detail::launch_option
{
  static constexpr bool needs_attribute_space        = true;
  static constexpr bool is_relevant_on_device        = false;
  static constexpr __detail::launch_option_kind kind = __detail::launch_option_kind::launch_priority;
  using __tag                                        = __launch_priority_t;
  const int priority                                 = 0;

  constexpr launch_priority_option(int __p) noexcept
      : priority(__p)
  {}

  constexpr _CCCL_HOST_DEVICE launch_priority_option(const launch_priority_option& __other) noexcept
      : priority(__other.priority)
  {}

  template <typename _Dimensions, typename... _Options>
  friend cudaError_t __detail::apply_kernel_config(
    const kernel_config<_Dimensions, _Options...>& __config, cudaLaunchConfig_t& __cuda_config, void* __kernel) noexcept;

private:
  [[nodiscard]] cudaError_t apply(cudaLaunchConfig_t& __config, void*) const noexcept
  {
    cudaLaunchAttribute __attr;
    __attr.id           = cudaLaunchAttributePriority;
    __attr.val.priority = priority;

    __config.attrs[__config.numAttrs++] = __attr;

    return cudaSuccess;
  }
};

struct __launch_priority_t
{
  constexpr launch_priority_option operator()(int __priority) const noexcept
  {
    return launch_priority_option(__priority);
  }
};

inline constexpr __launch_priority_t launch_priority;

template <typename... _OptionsToFilter>
struct __filter_options
{
  template <bool _Pred, typename _Option>
  [[nodiscard]] auto __option_or_empty(const _Option& __option)
  {
    if constexpr (_Pred)
    {
      return ::cuda::std::tuple(__option);
    }
    else
    {
      return ::cuda::std::tuple{};
    }
  }

  template <typename... _Options>
  [[nodiscard]] auto operator()(const _Options&... __options)
  {
    return ::cuda::std::tuple_cat(
      __option_or_empty<!__detail::__option_present_in_list<_Options, _OptionsToFilter...>>(__options)...);
  }
};

template <typename _Dimensions, typename... _Options>
auto __make_config_from_tuple(const _Dimensions& __dims, const ::cuda::std::tuple<_Options...>& __opts);

template <typename _T>
inline constexpr bool __is_kernel_config = false;

template <typename _Dimensions, typename... _Options>
inline constexpr bool __is_kernel_config<kernel_config<_Dimensions, _Options...>> = true;

template <typename _Tp>
_CCCL_CONCEPT __kernel_has_default_config =
  _CCCL_REQUIRES_EXPR((_Tp), _Tp& __t)(requires(__is_kernel_config<decltype(__t.default_config())>));

/**
 * @brief Type describing a kernel launch configuration
 *
 * This type should not be constructed directly and make_config helper function should be used instead
 *
 * @tparam Dimensions
 * cuda::experimental::hierarchy_dimensions instance that describes dimensions of thread hierarchy in this
 * configuration object
 *
 * @tparam Options
 * Types of options that were added to this configuration object
 */
template <typename _Dimensions, typename... _Options>
struct kernel_config : public __launch_option_wrapper<_Options>...
{
  _Dimensions dims;

  static_assert(::cuda::std::_And<::cuda::std::is_base_of<__detail::launch_option, _Options>...>::value);
  static_assert(__detail::no_duplicate_options<_Options...>);

  constexpr kernel_config(const _Dimensions& __dims, const _Options&... __opts)
      : __launch_option_wrapper<_Options>(__opts)...
      , dims(__dims)
  {}

  using __launch_option_wrapper<_Options>::query...;

  /**
   * @brief Add a new option to this configuration
   *
   * Returns a new kernel_config that has all option and dimensions from this kernel_config
   * with the option from the argument added to it
   *
   * @param new_option
   * Option to be added to the configuration
   */
  template <typename... _NewOptions>
  [[nodiscard]] auto add(const _NewOptions&... __new_options) const
  {
    return kernel_config<_Dimensions, _Options..., _NewOptions...>(
      dims, __launch_option_wrapper<_Options>::__option_..., __new_options...);
  }

  /**
   * @brief Combine this configuration with another configuration object
   *
   * Returns a new `kernel_config` that is a combination of this configuration and the configuration from argument.
   * It contains dimensions that are combination of dimensions in this object and the other configuration. The resulting
   * hierarchy holds levels present in both hierarchies. In case of overlap of levels hierarchy from this configuration
   * is prioritized, so the result always holds all levels from this hierarchy and non-overlapping
   * levels from the other hierarchy. This behavior is the same as `combine()` member function of the hierarchy type.
   * The result also contains configuration options from both configurations. In case the same type of a configuration
   * option is present in both configuration this configuration is copied into the resulting configuration.
   *
   * @param __other_config
   * Other configuration to combine with this configuration
   */
  template <typename _OtherDimensions, typename... _OtherOptions>
  [[nodiscard]] auto combine(const kernel_config<_OtherDimensions, _OtherOptions...>& __other_config) const
  {
    // can't use fully qualified kernel_config name here because of nvcc bug, TODO remove __make_config_from_tuple once
    // fixed
    return __make_config_from_tuple(::cuda::std::tuple_cat(
      ::cuda::std::make_tuple(dims.combine(__other_config.dims)),
      ::cuda::std::make_tuple(__launch_option_wrapper<_Options>::__option_...),
      __filter_options<_Options...>()(
        static_cast<__launch_option_wrapper<_OtherOptions>>(__other_config).__option_...)));
  }

  /**
   * @brief Combine this configuration with default configuration of a kernel functor
   *
   * Returns a new `kernel_config` that is a combination of this configuration and a default configuration from the
   * kernel argument. Default configuration is a `kernel_config` object returned from `default_config()` member function
   * of the kernel type. The configurations are combined using the `combine()` member function of this configuration.
   * If the kernel has no default configuration, a copy of this configuration is returned without any changes.
   *
   * @param __kernel
   * Kernel functor to search for the default configuration
   */
  template <typename _Kernel>
  [[nodiscard]] auto combine_with_default(const _Kernel& __kernel) const
  {
    if constexpr (__kernel_has_default_config<_Kernel>)
    {
      return combine(__kernel.default_config());
    }
    else
    {
      return *this;
    }
  }
};

// We can consider removing the operator&, but its convenient for in-line construction
template <typename _Dimensions, typename... _Options, typename _NewLevel>
_CCCL_HOST_API constexpr auto
operator&(const kernel_config<_Dimensions, _Options...>& __config, const _NewLevel& __new_level) noexcept
{
  return kernel_config(hierarchy_add_level(__config.dims, __new_level),
                       static_cast<__launch_option_wrapper<_Options>>(__config).__option_...);
}

template <typename _NewLevel, typename _Dimensions, typename... _Options>
_CCCL_HOST_API constexpr auto
operator&(const _NewLevel& __new_level, const kernel_config<_Dimensions, _Options...>& __config) noexcept
{
  return kernel_config(hierarchy_add_level(__config.dims, __new_level),
                       static_cast<__launch_option_wrapper<_Options>>(__config).__option_...);
}

template <typename _L1, typename _Dims1, typename _L2, typename _Dims2>
_CCCL_HOST_API constexpr auto
operator&(const level_dimensions<_L1, _Dims1>& __l1, const level_dimensions<_L2, _Dims2>& __l2) noexcept
{
  return kernel_config(make_hierarchy(__l1, __l2));
}

template <typename _Dimensions, typename... _Options>
auto __make_config_from_tuple(const ::cuda::std::tuple<_Dimensions, _Options...>& __opts)
{
  return ::cuda::std::make_from_tuple<kernel_config<_Dimensions, _Options...>>(__opts);
}

template <typename _Dimensions,
          typename... _Options,
          typename _Option,
          typename = ::cuda::std::enable_if_t<::cuda::std::is_base_of_v<__detail::launch_option, _Option>>>
[[nodiscard]] constexpr auto
operator&(const kernel_config<_Dimensions, _Options...>& __config, const _Option& __option) noexcept
{
  return __config.add(__option);
}

template <typename... _Levels,
          typename _Option,
          typename = ::cuda::std::enable_if_t<::cuda::std::is_base_of_v<__detail::launch_option, _Option>>>
[[nodiscard]] constexpr auto operator&(const hierarchy_dimensions<_Levels...>& __dims, const _Option& __option) noexcept
{
  return kernel_config(__dims, __option);
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
template <typename _BottomUnit, typename... _Levels, typename... _Opts>
[[nodiscard]] constexpr auto
make_config(const hierarchy_dimensions<_BottomUnit, _Levels...>& __dims, const _Opts&... __opts) noexcept
{
  return kernel_config<hierarchy_dimensions<_BottomUnit, _Levels...>, _Opts...>(__dims, __opts...);
}

/**
 * @brief A shorthand for creating a kernel configuration with a hierarchy of CUDA threads evenly
 * distributing elements among blocks and threads.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 * using namespace cuda::experimental;
 *
 * constexpr int threadsPerBlock = 256;
 * auto dims = distribute<threadsPerBlock>(numElements);
 *
 * // Equivalent to:
 * constexpr int threadsPerBlock = 256;
 * int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
 * auto dims = make_hierarchy(grid_dims(blocksPerGrid), block_dims<threadsPerBlock>());
 * @endcode
 */
template <int _ThreadsPerBlock>
constexpr auto distribute(int __numElements) noexcept
{
  int __blocksPerGrid = (__numElements + _ThreadsPerBlock - 1) / _ThreadsPerBlock;
  return make_config(make_hierarchy(grid_dims(__blocksPerGrid), block_dims<_ThreadsPerBlock>()));
}

template <typename... _Prev>
[[nodiscard]] constexpr auto __process_config_args(const ::cuda::std::tuple<_Prev...>& __previous)
{
  if constexpr (sizeof...(_Prev) == 0)
  {
    return kernel_config<__empty_hierarchy>(__empty_hierarchy());
  }
  else
  {
    return kernel_config(::cuda::std::apply(make_hierarchy<void, const _Prev&...>, __previous));
  }
}

template <typename... _Prev, typename _Arg, typename... _Rest>
[[nodiscard]] constexpr auto
__process_config_args(const ::cuda::std::tuple<_Prev...>& __previous, const _Arg& __arg, const _Rest&... __rest)
{
  if constexpr (::cuda::std::is_base_of_v<__detail::launch_option, _Arg>)
  {
    static_assert((::cuda::std::is_base_of_v<__detail::launch_option, _Rest> && ...),
                  "Hierarchy levels and launch options can't be mixed");
    if constexpr (sizeof...(_Prev) == 0)
    {
      return kernel_config(__empty_hierarchy(), __arg, __rest...);
    }
    else
    {
      return kernel_config(::cuda::std::apply(make_hierarchy<void, const _Prev&...>, __previous), __arg, __rest...);
    }
  }
  else
  {
    return __process_config_args(::cuda::std::tuple_cat(__previous, ::cuda::std::make_tuple(__arg)), __rest...);
  }
}

template <typename... _Args>
[[nodiscard]] constexpr auto make_config(const _Args&... __args)
{
  return __process_config_args(::cuda::std::make_tuple(), __args...);
}

namespace __detail
{

template <typename _Dimensions, typename... _Options>
inline unsigned int constexpr kernel_config_count_attr_space(const kernel_config<_Dimensions, _Options...>&) noexcept
{
  return (0 + ... + _Options::needs_attribute_space);
}

template <typename _Dimensions, typename... _Options>
[[nodiscard]] cudaError_t apply_kernel_config(
  const kernel_config<_Dimensions, _Options...>& __config,
  cudaLaunchConfig_t& __cuda_config,
  [[maybe_unused]] void* __kernel) noexcept
{
  cudaError_t __status = cudaSuccess;

  // Use short-cutting && to skip the rest on error, is this too convoluted?
  (void) (... && [&](cudaError_t __call_status) {
    __status = __call_status;
    return __call_status == cudaSuccess;
  }(static_cast<__launch_option_wrapper<_Options>>(__config).__option_.apply(__cuda_config, __kernel)));

  return __status;
}

// Needs to be a char casted to the appropriate type, if it would be a template
//  different instantiations would clash the extern symbol
[[nodiscard]] _CCCL_DEVICE static char* get_smem_ptr() noexcept
{
  extern __shared__ char dynamic_smem[];

  return &dynamic_smem[0];
}
} // namespace __detail

// Might consider cutting this one due to being a potential trap with missing & in auto& var = dynamic_smem_ref(...);
/**
 * @brief Returns a reference to shared memory variable in dynamic shared memory
 *
 * This function returns a reference to a variable placed in dynamic shared memory.
 * It accepts a kernel_config containing a dynamic_shared_memory_option.
 * Its only usable when dynamic shared memory option is holding a single object.
 */
template <typename _Dimensions, typename... _Options>
_CCCL_DEVICE auto& dynamic_smem_ref(const kernel_config<_Dimensions, _Options...>& __config) noexcept
{
  static_assert(_CUDA_STD_EXEC::__queryable_with<decltype(__config),
                                                 __dynamic_shared_memory_t<void, ::cuda::std::dynamic_extent, false>>,
                "Dynamic shared memory option not found in the kernel configuration");
  using __option_type = decltype(__config.query(__dynamic_shared_memory_t<void, ::cuda::std::dynamic_extent, false>{}));
  static_assert(__option_type::extent == 1, "Usable only on dynamic shared memory with a single element");

  return *reinterpret_cast<typename __option_type::content_type*>(__detail::get_smem_ptr());
}

/**
 * @brief Returns a cuda::std::span object referring to dynamic shared memory region
 *
 * This function returns a std::std::span object referring to the dynamic shared memory region
 * configured when launching the kernel.
 * It accepts a kernel_config containing a dynamic_shared_memory_option.
 * It is typed and sized according to the launch option provided as input.
 */
template <typename _Dimensions, typename... _Options>
_CCCL_DEVICE auto dynamic_smem_span(const kernel_config<_Dimensions, _Options...>& __config) noexcept
{
  static_assert(_CUDA_STD_EXEC::__queryable_with<decltype(__config),
                                                 __dynamic_shared_memory_t<void, ::cuda::std::dynamic_extent, false>>,
                "Dynamic shared memory option not found in the kernel configuration");
  auto __option       = __config.query(__dynamic_shared_memory_t<void, ::cuda::std::dynamic_extent, false>{});
  using __option_type = decltype(__option);

  return cuda::std::span<typename __option_type::content_type, __option_type::extent>(
    reinterpret_cast<typename __option_type::content_type*>(__detail::get_smem_ptr()), __option.size);
}

} // namespace cuda::experimental
#endif // _CCCL_STD_VER >= 2017

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__LAUNCH_CONFIGURATION
