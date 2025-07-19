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
  if (ctx)
  {
    *ctx = new stf_ctx_handle_t{context{}};
  }
}

void stf_ctx_finalize(stf_ctx_handle ctx)
{
  assert(ctx);
  delete ctx;
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

void stf_task_create(stf_ctx_handle ctx, stf_task_handle* t)
{
  assert(t);
  assert(ctx);

  *t = new stf_task_handle_t{ctx->ctx.task()};
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

void stf_task_destroy(stf_task_handle t)
{
  assert(t);
  delete t;
}
}
