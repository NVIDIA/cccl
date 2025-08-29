#include <cccl/c/experimental/stf/stf.h>
// #include <cccl/c/parallel/include/cccl/c/extern_c.h>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

extern "C" {

struct stf_ctx_handle_t
{
  context ctx;
};

struct stf_logical_data_handle_t
{
  // XXX should we always store a logical_data<slice<char>> instead ?
  logical_data_untyped ld;
};

struct stf_task_handle_t
{
  context::unified_task<> t;
};

void stf_ctx_create(stf_ctx_handle* ctx)
{
  assert(ctx);
  *ctx = new stf_ctx_handle_t{context{}};
}

void stf_ctx_create_graph(stf_ctx_handle* ctx)
{
  assert(ctx);
  *ctx = new stf_ctx_handle_t{context{graph_ctx()}};
}

void stf_ctx_finalize(stf_ctx_handle ctx)
{
  ctx->ctx.finalize();
  assert(ctx);
  delete ctx;
}

cudaStream_t stf_fence(stf_ctx_handle ctx)
{
  assert(ctx);
  return ctx->ctx.fence();
}

void stf_logical_data(stf_ctx_handle ctx, stf_logical_data_handle* ld, void* addr, size_t sz)
{
  assert(ld);
  assert(ctx);

  // Create a slice<char> logical data
  auto ld_typed = ctx->ctx.logical_data(make_slice((char*) addr, sz));

  // Stored in its untyped version
  *ld = new stf_logical_data_handle_t{ld_typed};
}

void stf_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol)
{
  assert(ld);
  ld->ld.set_symbol(symbol);
}

void stf_logical_data_destroy(stf_logical_data_handle ld)
{
  assert(ld);
  delete ld;
}

void stf_logical_data_empty(stf_ctx_handle ctx, size_t length, stf_logical_data_handle* to)
{
  assert(ctx);
  assert(to);

  auto ld_typed = ctx->ctx.logical_data(shape_of<slice<char>>(length));
  *to           = new stf_logical_data_handle_t{ld_typed};
}

// void stf_logical_data_like_empty(stf_ctx_handle ctx, const stf_logical_data_handle from, stf_logical_data_handle* to)
// {
//   assert(ctx);
//   assert(from);
//   assert(to);
//
//   auto ld_typed = ctx->ctx.logical_data(from->ld.shape());
//
//   // Stored in its untyped version
//   *to = new stf_logical_data_handle_t{ld_typed};
// }

void stf_token(stf_ctx_handle ctx, stf_logical_data_handle* ld)
{
  assert(ctx);
  assert(ld);

  *ld = new stf_logical_data_handle_t{ctx->ctx.token()};
}

/* Convert the C-API stf_exec_place to a C++ exec_place object */
exec_place to_exec_place(struct stf_exec_place* exec_p)
{
  if (exec_p->kind == STF_EXEC_PLACE_HOST)
  {
    return exec_place::host();
  }

  assert(exec_p->kind == STF_EXEC_PLACE_DEVICE);
  return exec_place::device(exec_p->u.device.dev_id);
}

/* Convert the C-API stf_data_place to a C++ data_place object */
data_place to_data_place(struct stf_data_place* data_p)
{
  assert(data_p);

  if (data_p->kind == STF_DATA_PLACE_HOST)
  {
    return data_place::host();
  }

  if (data_p->kind == STF_DATA_PLACE_MANAGED)
  {
    return data_place::managed();
  }

  if (data_p->kind == STF_DATA_PLACE_AFFINE)
  {
    return data_place::affine();
  }

  assert(data_p->kind == STF_DATA_PLACE_DEVICE);
  return data_place::device(data_p->u.device.dev_id);
}

void stf_task_create(stf_ctx_handle ctx, stf_task_handle* t)
{
  assert(t);
  assert(ctx);

  *t = new stf_task_handle_t{ctx->ctx.task()};
}

void stf_task_set_exec_place(stf_task_handle t, struct stf_exec_place* exec_p)
{
  assert(t);
  t->t.set_exec_place(to_exec_place(exec_p));
}

void stf_task_set_symbol(stf_task_handle t, const char* symbol)
{
  assert(t);
  assert(symbol);

  t->t.set_symbol(symbol);
}

void stf_task_add_dep(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m)
{
  assert(t);
  assert(ld);

  t->t.add_deps(task_dep_untyped(ld->ld, access_mode(m)));
}

void stf_task_add_dep_with_dplace(
  stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, struct stf_data_place* data_p)
{
  assert(t);
  assert(ld);
  assert(data_p);

  t->t.add_deps(task_dep_untyped(ld->ld, access_mode(m), to_data_place(data_p)));
}

void* stf_task_get(stf_task_handle t, int index)
{
  assert(t);
  auto s = t->t.template get<slice<const char>>(index);
  return (void*) s.data_handle();
}

void stf_task_start(stf_task_handle t)
{
  assert(t);
  t->t.start();
}

void stf_task_end(stf_task_handle t)
{
  assert(t);
  t->t.end();
}

void stf_task_enable_capture(stf_task_handle t)
{
  assert(t);
  t->t.enable_capture();
}

CUstream stf_task_get_custream(stf_task_handle t)
{
  assert(t);
  return (CUstream) t->t.get_stream();
}

void stf_task_destroy(stf_task_handle t)
{
  assert(t);
  delete t;
}

/**
 * Low level example of cuda_kernel(_chain)
 *   auto t = ctx.cuda_kernel_chain();
     t.add_deps(lX.read());
     t.add_deps(lY.rw());
     t->*[&]() {
     auto dX = t.template get<slice<double>>(0);
     auto dY = t.template get<slice<double>>(1);
     return std::vector<cuda_kernel_desc> {
         { axpy, 16, 128, 0, alpha, dX, dY },
         { axpy, 16, 128, 0, beta, dX, dY },
         { axpy, 16, 128, 0, gamma, dX, dY }
     };
  };

 *
 */
struct stf_cuda_kernel_handle_t
{
  // return type of ctx.cuda_kernel()
  using kernel_type = decltype(::std::declval<context>().cuda_kernel());
  kernel_type k;
};

void stf_cuda_kernel_create(stf_ctx_handle ctx, stf_cuda_kernel_handle* k)
{
  assert(k);
  assert(ctx);

  *k = new stf_cuda_kernel_handle_t{ctx->ctx.cuda_kernel()};
}

void stf_cuda_kernel_set_exec_place(stf_cuda_kernel_handle k, struct stf_exec_place* exec_p)
{
  assert(k);
  k->k.set_exec_place(to_exec_place(exec_p));
}

void stf_cuda_kernel_set_symbol(stf_cuda_kernel_handle k, const char* symbol)
{
  assert(k);
  assert(symbol);

  k->k.set_symbol(symbol);
}

void stf_cuda_kernel_add_dep(stf_cuda_kernel_handle k, stf_logical_data_handle ld, stf_access_mode m)
{
  assert(k);
  assert(ld);

  k->k.add_deps(task_dep_untyped(ld->ld, access_mode(m)));
}

void stf_cuda_kernel_start(stf_cuda_kernel_handle k)
{
  assert(k);
  k->k.start();
}

#if 0
//
//  template <typename Fun>
//  void configure_raw(Fun func, dim3 gridDim_, dim3 blockDim_, size_t sharedMem_, int arg_cnt, const void** args)
void stf_cuda_kernel_add_desc(stf_cuda_kernel_handle k, const void *func, dim3 gridDim_, dim3 blockDim_, size_t sharedMem_, int arg_cnt, const void** args)
{
    /* We convert the function to a CUfunction because this code is a shared
     * library which cannot launch kernels using cudaLaunchKernel directly, or we
     * will get invalid device function. */
    //CUfunction cufunc;
    //cudaGetFuncBySymbol(&cufunc, (void *)func);
    CUkernel cukernel;
    cudaGetKernel(&cukernel, (void *)func);

    cuda_kernel_desc desc;
    desc.configure_raw(cukernel, gridDim_, blockDim_, sharedMem_, arg_cnt, args);

    k->k.add_kernel_desc(mv(desc));
}
#endif

void stf_cuda_kernel_add_desc_cufunc(
  stf_cuda_kernel_handle k,
  CUfunction cufunc,
  dim3 gridDim_,
  dim3 blockDim_,
  size_t sharedMem_,
  int arg_cnt,
  const void** args)
{
  cuda_kernel_desc desc;
  desc.configure_raw(cufunc, gridDim_, blockDim_, sharedMem_, arg_cnt, args);

  k->k.add_kernel_desc(mv(desc));
}

void* stf_cuda_kernel_get_arg(stf_cuda_kernel_handle k, int index)
{
  auto s = k->k.template get<slice<const char>>(index);
  return (void*) s.data_handle();
}

void stf_cuda_kernel_end(stf_cuda_kernel_handle k)
{
  assert(k);
  k->k.end();
}

void stf_cuda_kernel_destroy(stf_cuda_kernel_handle t)
{
  assert(t);
  delete t;
}
}
