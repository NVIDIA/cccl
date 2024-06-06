#define cudaLaunchKernelEx cudaLaunchKernelExTestReplacement
#include <cuda/experimental/launch.cuh>
#undef cudaLaunchKernelEx

#include "../hierarchy/testing_common.cuh"

static cudaLaunchConfig_t expectedConfig;

template <typename... ExpTypes, typename... ActTypes>
cudaError_t
cudaLaunchKernelExTestReplacement(const cudaLaunchConfig_t* config, void (*kernel)(ExpTypes...), ActTypes&&... args)
{
  REQUIRE(expectedConfig.numAttrs == config->numAttrs);

  return cudaLaunchKernelEx(config, kernel, cuda::std::forward<ActTypes>(args)...);
}

__global__ void empty_kernel(int i) {}

TEST_CASE("Launch configuration", "[launch]")
{
  cudaLaunchAttribute attrs[32];

  auto dims = cudax::make_hierarchy(cudax::block_dims<256>(), cudax::grid_dims(4));

  expectedConfig                          = {0};
  expectedConfig.numAttrs                 = 1;
  expectedConfig.attrs                    = &attrs[0];
  expectedConfig.attrs[0].id              = cudaLaunchAttributeCooperative;
  expectedConfig.attrs[0].val.cooperative = 1;
  auto config                             = cudax::make_config(dims, cudax::cooperative_launch());
  cudax::launch(config, empty_kernel, 1);
}
