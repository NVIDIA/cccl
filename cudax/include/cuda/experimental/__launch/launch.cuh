#ifndef _CUDAX__LAUNCH_LAUNCH
#define _CUDAX__LAUNCH_LAUNCH
#include <cuda_runtime.h>

#include <cuda/experimental/__launch/confiuration.cuh>
#include <cuda/std/__exception/cuda_error.h>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

namespace detail
{
template <typename Config, typename Kernel, class... Args>
__global__ void kernel_launcher(const Config conf, Kernel kernel_fn, Args... args)
{
  kernel_fn(conf, args...);
}

template <typename Kernel, class... Args>
__global__ void kernel_launcher_no_config(Kernel kernel_fn, Args... args)
{
  kernel_fn(args...);
}

template <typename Config, typename Kernel, typename... Args>
cudaError_t launch_impl(Config conf, const Kernel& kernel_fn, const Args&... args) noexcept
{
  cudaLaunchConfig_t config               = {0};
  cudaError_t status                      = cudaSuccess;
  constexpr bool has_cluster_level        = has_level<cluster_level, decltype(conf.dims)>;
  constexpr unsigned int num_attrs_needed = conf.num_attrs_needed() + has_cluster_level + 1;
  cudaLaunchAttribute attrs[num_attrs_needed];
  config.attrs    = &attrs[0];
  config.numAttrs = 0;

  status = conf.apply(config, reinterpret_cast<void*>(kernel_fn));
  if (status != cudaSuccess)
  {
    return status;
  }

  // auto conf = transform_config(conf, kernel_fn);

  config.blockDim = conf.dims.extents(thread, block);
  config.gridDim  = conf.dims.extents(block, grid);

  if constexpr (has_cluster_level)
  {
    dim3 cluster_dims                            = conf.dims.extents(block, cluster);
    config.attrs[config.numAttrs].id             = cudaLaunchAttributeClusterDimension;
    config.attrs[config.numAttrs].val.clusterDim = {cluster_dims.x, cluster_dims.y, cluster_dims.z};
    config.numAttrs++;
  }

  // TODO lower to cudaLaunchKernelExC?
  return cudaLaunchKernelEx(&config, kernel_fn, args...);
}
} // namespace detail

template <typename... Args, typename Config, typename Dimensions, typename Kernel>
void launch(const kernel_config<Dimensions, Config>& conf, const Kernel& kernel, const Args&... args)
{
  cudaError_t status;
  if constexpr (cuda::std::is_invocable_v<Kernel, kernel_config<Dimensions, Config>, Args...>)
  {
    auto launcher = detail::kernel_launcher<kernel_config<Dimensions, Config>, Kernel, Args...>;
    status        = detail::launch_impl(conf, launcher, conf, kernel, args...);
  }
  else
  {
    auto launcher =
      detail::kernel_launcher_no_config<std::remove_reference_t<Kernel>, std::remove_reference_t<Args>...>;
    status = detail::launch_impl(conf, launcher, kernel, args...);
  }
  if (status != cudaSuccess)
  {
    throw ::cuda::cuda_error(status, "Failed to launch a kernel");
  }
}

template <typename... Args, typename... Levels, typename Kernel>
void launch(const hierarchy_dimensions<Levels...>& dims, const Kernel& kernel, const Args&... args)
{
  cudaError_t status;
  if constexpr (cuda::std::is_invocable_v<Kernel, hierarchy_dimensions<Levels...>, Args...>)
  {
    auto launcher = detail::kernel_launcher<hierarchy_dimensions<Levels...>, Kernel, Args...>;
    status        = detail::launch_impl(kernel_config(dims), launcher, dims, kernel, args...);
  }
  else
  {
    auto launcher = detail::kernel_launcher_no_config<Kernel, Args...>;
    status        = detail::launch_impl(kernel_config(dims), launcher, kernel, args...);
  }
  if (status != cudaSuccess)
  {
    throw ::cuda::cuda_error(status, "Failed to launch a kernel");
  }
}

/* Functions accepting __global__ function pointer (needs to be instantiated or template arguments need to be passed
 * into launch template), but it will support implicit conversion of arguments */
template <typename... ExpArgs, typename... ActArgs, typename Config, typename Dimensions>
void launch(const kernel_config<Dimensions, Config>& conf,
            void (*kernel)(kernel_config<Dimensions, Config>, ExpArgs...),
            ActArgs&&... actArgs)
{
  cudaError_t status = [&](ExpArgs... args) {
    return detail::launch_impl(conf, kernel, conf, args...);
  }(std::forward<ActArgs>(actArgs)...);
  if (status != cudaSuccess)
  {
    throw ::cuda::cuda_error(status, "Failed to launch a kernel");
  }
}

template <typename... ExpArgs, typename... ActArgs, typename... Levels>
void launch(const hierarchy_dimensions<Levels...>& dims,
            void (*kernel)(hierarchy_dimensions<Levels...>, ExpArgs...),
            ActArgs&&... actArgs)
{
  cudaError_t status = [&](ExpArgs... args) {
    return detail::launch_impl(kernel_config(dims), kernel, dims, args...);
  }(std::forward<ActArgs>(actArgs)...);
  if (status != cudaSuccess)
  {
    throw ::cuda::cuda_error(status, "Failed to launch a kernel");
  }
}

template <typename... ExpArgs, typename... ActArgs, typename Config, typename Dimensions>
void launch(const kernel_config<Dimensions, Config>& conf, void (*kernel)(ExpArgs...), ActArgs&&... actArgs)
{
  cudaError_t status = [&](ExpArgs... args) {
    return detail::launch_impl(conf, kernel, args...);
  }(std::forward<ActArgs>(actArgs)...);
  if (status != cudaSuccess)
  {
    throw ::cuda::cuda_error(status, "Failed to launch a kernel");
  }
}

template <typename... ExpArgs, typename... ActArgs, typename... Levels>
void launch(const hierarchy_dimensions<Levels...>& dims, void (*kernel)(ExpArgs...), ActArgs&&... actArgs)
{
  cudaError_t status = [&](ExpArgs... args) {
    return detail::launch_impl(kernel_config(dims), kernel, args...);
  }(std::forward<ActArgs>(actArgs)...);
  if (status != cudaSuccess)
  {
    throw ::cuda::cuda_error(status, "Failed to launch a kernel");
  }
}

} // namespace cuda::experimental
#endif // _CCCL_STD_VER >= 2017
#endif // _CUDAX__LAUNCH_LAUNCH
