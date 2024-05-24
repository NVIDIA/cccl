#ifndef __CUDA_LAUNCH_CONFIGURATION__
#define __CUDA_LAUNCH_CONFIGURATION__
#include <cuda/experimental/hierarchy.cuh>
#include <cuda/std/span>
#include <cuda/std/tuple>

namespace cuda::experimental
{

struct sm_60
{};
struct sm_70
{};
struct sm_80
{};
struct sm_90
{};

namespace detail
{
struct launch_option
{
  static constexpr bool needs_attribute_space = false;
  static constexpr bool is_relevant_on_device = false;

  template <typename KernelFn>
  cudaError_t apply(cudaLaunchConfig_t& config, KernelFn kernel) const noexcept
  {
    return cudaSuccess;
  }
};
} // namespace detail

/*
template <typename... Opts1, typename... Opts2>
auto constexpr operator&(const config_list<Opts1...>& opts1, const config_list<Opts2...>& opts2) noexcept
{
  static_assert(std::disjunction_v<std::is_base_of<launch_option, Opts1>...>);
  static_assert(std::disjunction_v<std::is_base_of<launch_option, Opts2>...>);
  return config_list<Opts1..., Opts2...>(static_cast<Opts1>(opts1)..., static_cast<Opts2>(opts2)...);
}

*/

struct launch_on_option : public detail::launch_option
{
  // cudaStream_t until we have a proper stream abstraction
  cudaStream_t stream;

  launch_on_option(cudaStream_t stream)
      : stream(stream){};

  template <typename KernelFn>
  cudaError_t apply(cudaLaunchConfig_t& config, KernelFn kernel) const noexcept
  {
    config.stream = stream;

    return cudaSuccess;
  }
};

static launch_on_option launch_on(cudaStream_t stream)
{
  return launch_on_option(stream);
}

struct cooperative_launch_option : public detail::launch_option
{
  static constexpr bool needs_attribute_space = true;
  static constexpr bool is_relevant_on_device = true;

  // Probably private
  constexpr cooperative_launch_option() = default;

  template <typename KernelFn>
  cudaError_t apply(cudaLaunchConfig_t& config, KernelFn kernel) const noexcept
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
struct dyn_smem_config_element : public detail::launch_option
{
  const std::size_t size                      = Extent;
  static constexpr bool is_relevant_on_device = true;

  // Probably private
  constexpr dyn_smem_config_element(std::size_t set_size) noexcept
      : size(set_size)
  {}

  template <typename KernelFn>
  cudaError_t apply(cudaLaunchConfig_t& config, KernelFn kernel) const noexcept
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

// Functions to create dyn_smem_config_element, should all config elements be created with a function?
template <typename Content, std::size_t Extent = 1>
constexpr dyn_smem_config_element<Content, Extent> dynamic_shared_memory() noexcept
{
  static_assert(Extent != cuda::std::dynamic_extent, "Size needs to be provided when dynamic_extent is specified");

  return dyn_smem_config_element<Content, Extent>(Extent);
}

template <typename Content>
constexpr dyn_smem_config_element<Content, cuda::std::dynamic_extent> dynamic_shared_memory(std::size_t size) noexcept
{
  return dyn_smem_config_element<Content, cuda::std::dynamic_extent>(size);
}

struct launch_priority_config_element : public detail::launch_option
{
  static constexpr bool needs_attribute_space = true;
  static constexpr bool is_relevant_on_device = false;
  unsigned int priority;

  launch_priority_config_element(unsigned int p) noexcept
      : priority(p)
  {}

  template <typename KernelFn>
  cudaError_t apply(cudaLaunchConfig_t& config, KernelFn kernel) const noexcept
  {
    cudaLaunchAttribute attr;
    attr.id           = cudaLaunchAttributePriority;
    attr.val.priority = priority;

    config.attrs[config.numAttrs++] = attr;

    return cudaSuccess;
  }
};

launch_priority_config_element launch_priority(unsigned int p) noexcept
{
  return launch_priority_config_element(p);
}

template <typename Dimensions, typename... Options>
struct kernel_config : public Options...
{
  Dimensions dims;

  static_assert(cuda::std::disjunction_v<std::true_type, cuda::std::is_base_of<detail::launch_option, Options>...>);

  constexpr kernel_config(const Dimensions& dims, const Options&... opts) noexcept
      : Options(opts)...
      , dims(dims){};

  static unsigned int constexpr num_attrs_needed() noexcept
  {
    return (0 + ... + Options::needs_attribute_space);
  }

  // KernelFn can probably be void* to avoid multiple instantiations
  template <typename KernelFn>
  cudaError_t apply(cudaLaunchConfig_t& config, KernelFn kernel) const noexcept
  {
    cudaError_t status = cudaSuccess;

    // Use short-cutting && to skip the rest on error, is this too convoluted?
    (void) (... && [&](cudaError_t call_status) {
      status = call_status;
      return status == cudaSuccess;
    }(Options::apply(config, kernel)));

    return status;
  }
};

template <typename... Levels,
          typename Option,
          typename = cuda::std::enable_if_t<cuda::std::is_base_of_v<detail::launch_option, Option>>>
auto constexpr operator&(const hierarchy_dimensions<Levels...>& dims, const Option& option) noexcept
{
  return kernel_config(dims, option);
}

template <typename... Levels, typename... Opts>
auto constexpr make_config(const hierarchy_dimensions<Levels...>& dims, const Opts&... opts) noexcept
{
  return kernel_config<hierarchy_dimensions<Levels...>, Opts...>(dims, opts...);
}

template <typename TargetArch, typename Config>
struct arch_specific_config
{
  using target_arch = TargetArch;
  Config conf;

  constexpr arch_specific_config(const Config& c) noexcept
      : conf(c)
  {}
  constexpr arch_specific_config(const TargetArch& arch, const Config& c) noexcept
      : conf(c)
  {}
};

template <typename Arch, typename Dims, typename Config>
constexpr auto operator>>=(Arch arch, const kernel_config<Dims, Config>& config) noexcept
{
  return arch_specific_config<Arch, kernel_config<Dims, Config>>(arch, config);
}

template <typename... Configs>
struct per_arch_kernel_config
{
  const cuda::std::tuple<Configs...> configs;

  constexpr per_arch_kernel_config(const Configs&... confs) noexcept
      : configs(confs...)
  {}
};

namespace detail
{
template <typename TargetArch>
struct get_target_config_helper
{
  template <typename Config, typename... Rest>
  constexpr auto& operator()(const Config& conf, const Rest&... rest) noexcept
  {
    if constexpr (std::is_same_v<TargetArch, typename Config::target_arch>)
    {
      return conf.conf;
    }
    else
    {
      return (*this)(rest...);
    }
  }
};
} // namespace detail

template <typename TargetArch, typename... Configs>
constexpr auto& get_target_config(const per_arch_kernel_config<Configs...>& config) noexcept
{
  return cuda::std::apply(detail::get_target_config_helper<TargetArch>{}, config.configs);
}

static char* __device__ get_smem_ptr() noexcept
{
  extern __shared__ char dynamic_smem[];

  return &dynamic_smem[0];
}

template <typename Content>
Content& __device__ dynamic_smem_ref(const dyn_smem_config_element<Content, 1>& m) noexcept
{
  return *reinterpret_cast<Content*>(get_smem_ptr());
}

template <typename Content, std::size_t Extent>
cuda::std::span<Content, Extent> __device__ dynamic_smem_span(const dyn_smem_config_element<Content, Extent>& m) noexcept
{
  return cuda::std::span<Content, Extent>(reinterpret_cast<Content*>(get_smem_ptr()), m.size);
}

} // namespace cuda::experimental
#endif
