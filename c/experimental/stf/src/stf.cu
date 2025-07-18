#include <cccl/c/experimental/stf/stf.h>
// #include <cccl/c/parallel/include/cccl/c/extern_c.h>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

extern "C" {

struct stf_ctx_handle
{
  context* ctx;
};

void stf_ctx_create(stf_ctx_handle* handle)
{
  return new context{};
}

void stf_ctx_finalize(stf_ctx_handle* handle)
{
  if (handle)
  {
    handle->finalize();
  }
}
}
