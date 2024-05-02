#ifndef __CUDA_LAUNCH__
#define __CUDA_LAUNCH__
#include <cuda_runtime.h>

#include <cuda/next/launch_confiuration.cuh>

namespace cuda::experimental
{

struct launchErrorException
{
  cudaError_t error;
};

namespace detail
{
template <typename Config, typename Kernel, class... Args>
__global__ void kernel_launcher(const Config conf, Kernel kernel_fn, Args... args)
{
  kernel_fn(conf, args...);
}

template <typename Config, typename Kernel, typename... Args>
cudaError_t launch_impl(Config conf, Kernel kernel_fn, Args&&... args) noexcept
{
  cudaLaunchConfig_t config               = {0};
  cudaError_t status                      = cudaSuccess;
  constexpr bool has_cluster_level        = has_level<cluster_level, decltype(conf.dims)>;
  constexpr unsigned int num_attrs_needed = conf.list.num_attrs_needed() + has_cluster_level;
  cudaLaunchAttribute attrs[num_attrs_needed];
  config.attrs    = &attrs[0];
  config.numAttrs = 0;

  config.blockDim = conf.dims.flatten(thread, block);
  config.gridDim  = conf.dims.flatten(block, grid);

  status = conf.list.apply(config, kernel_fn);
  if (status != cudaSuccess)
  {
    return status;
  }

  if constexpr (has_cluster_level)
  {
    dim3 cluster_dims                            = conf.dims.flatten(block, cluster);
    config.attrs[config.numAttrs].id             = cudaLaunchAttributeClusterDimension;
    config.attrs[config.numAttrs].val.clusterDim = {cluster_dims.x, cluster_dims.y, cluster_dims.z};
    config.numAttrs++;
  }

  return cudaLaunchKernelEx(&config, kernel_fn, std::forward<Args>(args)...);
}
} // namespace detail

/* Functions accepting a functor/lambda */
template <typename... Args, typename Config, typename Dimensions, typename Kernel>
void launch(cudaError_t& status, const kernel_config<Dimensions, Config>& conf, Kernel&& kernel, Args&&... args) noexcept
{
  auto launcher = detail::kernel_launcher<kernel_config<Dimensions, Config>,
                                          std::remove_reference_t<Kernel>,
                                          std::remove_reference_t<Args>...>;
  status        = detail::launch_impl(conf, launcher, conf, std::forward<Kernel>(kernel), std::forward<Args>(args)...);
}

template <typename... Args, typename Config, typename Dimensions, typename Kernel>
void launch(const kernel_config<Dimensions, Config>& conf, Kernel&& kernel, Args&&... args)
{
  cudaError_t status;
  launch(status, conf, std::forward<Kernel>(kernel), std::forward<Args>(args)...);
  if (status != cudaSuccess)
  {
    throw launchErrorException{status};
  }
}

// const ref to Args for now, had some issues with Args&&
template <typename... Args, typename... Levels, typename Kernel>
void launch(
  cudaError_t& status, const hierarchy_dimensions<Levels...>& dims, const Kernel& kernel, const Args&... args) noexcept
{
  auto launcher = detail::kernel_launcher<hierarchy_dimensions<Levels...>, Kernel, Args...>; //(dims, kernel, args...);
  status        = detail::launch_impl(kernel_config(dims), launcher, dims, kernel, args...);
}

template <typename... Args, typename... Levels, typename Kernel>
void launch(const hierarchy_dimensions<Levels...>& dims, Kernel kernel, const Args&... args)
{
  cudaError_t status;
  auto launcher =
    detail::kernel_launcher<hierarchy_dimensions<Levels...>, Kernel, std::remove_reference_t<Args>...>; //(dims, kernel,
                                                                                                        // args...);
  status = detail::launch_impl(kernel_config(dims), launcher, dims, std::forward<Kernel>(kernel), args...);
  if (status != cudaSuccess)
  {
    throw launchErrorException{status};
  }
}

/* Functions accepting __global__ function template that will instantiate it */
template <typename... Args, typename Config, typename Dimensions>
void launch(cudaError_t& status,
            const kernel_config<Dimensions, Config>& conf,
            void (*kernel)(kernel_config<Dimensions, Config>, std::remove_reference_t<Args>...),
            Args&&... args) noexcept
{
  status = detail::launch_impl(conf, kernel, conf, std::forward<Args>(args)...);
}

template <typename... Args, typename Config, typename Dimensions>
void launch(const kernel_config<Dimensions, Config>& conf,
            void (*kernel)(kernel_config<Dimensions, Config>, std::remove_reference_t<Args>...),
            Args&&... args)
{
  cudaError_t status = detail::launch_impl(conf, kernel, conf, std::forward<Args>(args)...);
  if (status != cudaSuccess)
  {
    throw launchErrorException{status};
  }
}

template <typename... Args, typename... Levels>
void launch(cudaError_t& status,
            const hierarchy_dimensions<Levels...>& dims,
            void (*kernel)(hierarchy_dimensions<Levels...>, std::remove_reference_t<Args>...),
            Args&&... args) noexcept
{
  status = detail::launch_impl(kernel_config(dims), kernel, dims, std::forward<Args>(args)...);
}

template <typename... Args, typename... Levels>
void launch(const hierarchy_dimensions<Levels...>& dims,
            void (*kernel)(hierarchy_dimensions<Levels...>, std::remove_reference_t<Args>...),
            Args&&... args)
{
  cudaError_t status = detail::launch_impl(kernel_config(dims), kernel, dims, std::forward<Args>(args)...);
  if (status != cudaSuccess)
  {
    throw launchErrorException{status};
  }
}

template <typename... Args, typename Config, typename Dimensions>
void launch(cudaError_t& status,
            const kernel_config<Dimensions, Config>& conf,
            void (*kernel)(std::remove_reference_t<Args>...),
            Args&&... args) noexcept
{
  status = detail::launch_impl(conf, kernel, std::forward<Args>(args)...);
}

template <typename... Args, typename Config, typename Dimensions>
void launch(const kernel_config<Dimensions, Config>& conf,
            void (*kernel)(std::remove_reference_t<Args>...),
            Args&&... args)
{
  cudaError_t status = detail::launch_impl(conf, kernel, std::forward<Args>(args)...);
  if (status != cudaSuccess)
  {
    throw launchErrorException{status};
  }
}

template <typename... Args, typename... Levels>
void launch(cudaError_t& status,
            const hierarchy_dimensions<Levels...>& dims,
            void (*kernel)(std::remove_reference_t<Args>...),
            Args... args) noexcept
{
  status = detail::launch_impl(kernel_config(dims), kernel, std::forward<Args>(args)...);
}

template <typename... Args, typename... Levels>
void launch(const hierarchy_dimensions<Levels...>& dims,
            void (*kernel)(std::remove_reference_t<Args>...),
            Args&&... args)
{
  cudaError_t status = detail::launch_impl(kernel_config(dims), kernel, std::forward<Args>(args)...);
  if (status != cudaSuccess)
  {
    throw launchErrorException{status};
  }
}

/* Functions accepting __global__ function pointer (needs to be instantiated or template arguments need to be passed
 * into launch template), but it will support implicit conversion of arguments */
template <typename... ExpArgs, typename... ActArgs, typename Config, typename Dimensions>
void launch(cudaError_t& status,
            const kernel_config<Dimensions, Config>& conf,
            void (*kernel)(kernel_config<Dimensions, Config>, std::remove_reference_t<ExpArgs>...),
            ActArgs&&... actArgs) noexcept
{
  status = [&](ExpArgs... args) {
    return detail::launch_impl(conf, kernel, conf, std::forward<ExpArgs>(args)...);
  }(std::forward<ActArgs>(actArgs)...);
}

template <typename... ExpArgs,
          typename... ActArgs,
          typename Config,
          typename Dimensions,
          typename = std::enable_if_t<
            !std::conjunction<typename std::is_same<ExpArgs, std::remove_reference_t<ActArgs>>...>::value>>
void launch(const kernel_config<Dimensions, Config>& conf,
            void (*kernel)(kernel_config<Dimensions, Config>, ExpArgs...),
            ActArgs&&... actArgs)
{
  cudaError_t status = [&](ExpArgs... args) {
    return detail::launch_impl(conf, kernel, conf, std::forward<ExpArgs>(args)...);
  }(std::forward<ActArgs>(actArgs)...);
  if (status != cudaSuccess)
  {
    throw launchErrorException{status};
  }
}

} // namespace cuda::experimental
#endif
