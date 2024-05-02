#ifndef __CUDA_LAUNCH_CONFIGURATION__
#define __CUDA_LAUNCH_CONFIGURATION__
#include <cuda/next/hierarchy_dimensions.cuh>
#include <cuda/std/span>
#include <cuda/std/tuple>

namespace cuda::experimental
{

struct architecture
{};

struct sm_60 : public architecture
{};
struct sm_70 : public architecture
{};
struct sm_80 : public architecture
{};
struct sm_90 : public architecture
{};

// template <typename Arch>
// concept DeviceArchitecture = std::is_base_of_v<architecture, Arch>;

struct launch_config_element
{
  static constexpr bool needs_attribute_space = false;

  template <typename KernelFn>
  cudaError_t apply(cudaLaunchConfig_t& config, KernelFn kernel) const noexcept
  {
    return cudaSuccess;
  }
};

// template <typename Element>
// concept LaunchConfigElement = std::is_base_of_v<launch_config_element, Element>;

template <typename... Extensions>
struct config_list : public Extensions...
{
  static_assert(std::disjunction_v<std::true_type, std::is_base_of<launch_config_element, Extensions>...>);

  constexpr config_list(const Extensions&... es) noexcept
      : Extensions(es)...
  {}

  static unsigned int constexpr num_attrs_needed() noexcept
  {
    return (0 + ... + Extensions::needs_attribute_space);
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
    }(Extensions::apply(config, kernel)));

    return status;
  }
};

template <typename... Es1, typename... Es2>
auto constexpr operator&(const config_list<Es1...>& es1, const config_list<Es2...>& es2) noexcept
{
  static_assert(std::disjunction_v<std::true_type, std::is_base_of<launch_config_element, Es1>...>);
  static_assert(std::disjunction_v<std::true_type, std::is_base_of<launch_config_element, Es2>...>);
  return config_list<Es1..., Es2...>(static_cast<Es1>(es1)..., static_cast<Es2>(es2)...);
}

struct cooperative_launch_config_element : public launch_config_element
{
  static constexpr bool needs_attribute_space = true;

  // Probably private
  constexpr cooperative_launch_config_element() = default;

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

constexpr config_list<cooperative_launch_config_element> cooperative_launch() noexcept
{
  return config_list(cooperative_launch_config_element());
}

template <typename Content>
struct workspace_config_element : public launch_config_element
{
  workspace_config_element(Content& c) noexcept
      : content(&c){};

  Content* content;
};

template <typename Content>
config_list<workspace_config_element<Content>> workspace(Content& c) noexcept
{
  return config_list(workspace_config_element<Content>(c));
}

template <typename Content, std::size_t Extent = 1>
struct dyn_smem_config_element : public launch_config_element
{
  const std::size_t size = Extent;

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
constexpr config_list<dyn_smem_config_element<Content, Extent>> dynamic_shared_memory() noexcept
{
  static_assert(Extent != cuda::std::dynamic_extent, "Size needs to be provided when dynamic_extent is specified");

  return config_list(dyn_smem_config_element<Content, Extent>(Extent));
}

template <typename Content>
constexpr config_list<dyn_smem_config_element<Content, cuda::std::dynamic_extent>>
dynamic_shared_memory(std::size_t size) noexcept
{
  return config_list(dyn_smem_config_element<Content, cuda::std::dynamic_extent>(size));
}

struct launch_priority_config_element : public launch_config_element
{
  static constexpr bool needs_attribute_space = true;
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

config_list<launch_priority_config_element> launch_priority(unsigned int p) noexcept
{
  return config_list(launch_priority_config_element(p));
}

template <typename Dimensions, typename ConfigList = config_list<>>
struct kernel_config
{
  const Dimensions dims;
  const ConfigList list;

  constexpr kernel_config(const Dimensions& ls) noexcept
      : dims(ls)
      , list(){};

  constexpr kernel_config(const Dimensions& ls, const ConfigList& cs) noexcept
      : dims(ls)
      , list(cs){};
};

template <typename... Levels, typename... Config>
auto constexpr operator&(const hierarchy_dimensions<Levels...>& levels, const config_list<Config...>& list) noexcept
{
  return kernel_config(levels, list);
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

template <typename Content>
Content& __device__ get_workspace_content(const workspace_config_element<Content>& w) noexcept
{
  return *w.content;
}

template <typename... Config>
auto& __device__ get_workspace_content(const kernel_config<Config...>& c) noexcept
{
  return get_workspace_content(c.list);
}

char* __device__ get_smem_ptr() noexcept
{
  extern __shared__ char dynamic_smem[];

  return &dynamic_smem[0];
}

template <typename Content>
Content& __device__ get_smem_content_ref(const dyn_smem_config_element<Content, 1>& m) noexcept
{
  return *reinterpret_cast<Content*>(get_smem_ptr());
}

template <typename... Config>
auto& __device__ get_smem_content_ref(const kernel_config<Config...>& c) noexcept
{
  return get_smem_content_ref(c.list);
}

template <typename Content, std::size_t Extent>
cuda::std::span<Content, Extent> __device__ get_smem_content(const dyn_smem_config_element<Content, Extent>& m) noexcept
{
  return cuda::std::span<Content, Extent>(reinterpret_cast<Content*>(get_smem_ptr()), m.size);
}

template <typename... Config>
auto __device__ get_smem_content(const kernel_config<Config...>& c) noexcept
{
  return get_smem_content(c.list);
}

} // namespace cuda::experimental
#endif
