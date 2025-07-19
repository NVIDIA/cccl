#include <cccl/c/experimental/stf/stf.h>
// #include <cccl/c/parallel/include/cccl/c/extern_c.h>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

extern "C" {

struct stf_ctx_handle_t
{
  context ctx;
};

void stf_ctx_create(stf_ctx_handle* ctx)
{
  if (ctx) {
    *ctx = new stf_ctx_handle_t{context{}};
  }
}

void stf_ctx_finalize(stf_ctx_handle ctx)
{
  delete ctx;
}

struct stf_logical_data_handle_t
{
  // XXX should we always store a logical_data<slice<char>> instead ?
  logical_data_untyped ld;
};

void stf_logical_data(stf_ctx_handle ctx, stf_logical_data_handle *ld, void *addr, size_t sz)
{
   assert(ld);
   assert(ctx);

   // Create a slice<char> logical data
   auto ld_typed = ctx->ctx.logical_data(make_slice((char *)addr, sz));

   // Stored in its untyped version
   *ld = new stf_logical_data_handle_t{ld_typed};
}

void stf_logical_data_destroy(stf_ctx_handle /* ctx */, stf_logical_data_handle ld)
{
    delete ld;
}

}
