#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

// TODO use CCCL_C_EXTERN_C_BEGIN/CCCL_C_EXTERN_C_END
#ifdef __cplusplus
extern "C" {
#endif

typedef enum stf_access_mode
{
  STF_NONE  = 0,
  STF_READ  = 1 << 0,
  STF_WRITE = 1 << 1,
  STF_RW    = STF_READ | STF_WRITE
} stf_access_mode;

typedef struct stf_ctx_handle_t* stf_ctx_handle;

void stf_ctx_create(stf_ctx_handle* ctx);
void stf_ctx_finalize(stf_ctx_handle ctx);

// TODO stf_ctx_set_mode() + define enum with GRAPH, STREAM, ...
// TODO stf_ctx_is_graph()

cudaStream_t stf_fence(stf_ctx_handle ctx);

typedef struct stf_logical_data_handle_t* stf_logical_data_handle;

void stf_logical_data(stf_ctx_handle ctx, stf_logical_data_handle* ld, void* addr, size_t sz);
void stf_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol);
void stf_logical_data_destroy(stf_logical_data_handle ld);

// TODO
// void stf_logical_data_wait(stf_logical_data_handle ld);

void stf_token(stf_ctx_handle ctx, stf_logical_data_handle* ld);

typedef struct stf_task_handle_t* stf_task_handle;

void stf_task_create(stf_ctx_handle ctx, stf_task_handle* t);
void stf_task_set_symbol(stf_task_handle t, const char* symbol);
void stf_task_add_dep(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m);
void stf_task_start(stf_task_handle t);
void stf_task_end(stf_task_handle t);
CUstream stf_task_get_custream(stf_task_handle t);
void* stf_task_get(stf_task_handle t, int submitted_index);
void stf_task_destroy(stf_task_handle t);

typedef struct stf_cuda_kernel_handle_t* stf_cuda_kernel_handle;

void stf_cuda_kernel_create(stf_ctx_handle ctx, stf_cuda_kernel_handle* k);
void stf_cuda_kernel_set_symbol(stf_cuda_kernel_handle k, const char* symbol);
void stf_cuda_kernel_add_dep(stf_cuda_kernel_handle k, stf_logical_data_handle ld, stf_access_mode m);
void stf_cuda_kernel_start(stf_cuda_kernel_handle k);

void stf_cuda_kernel_add_desc_cufunc(
  stf_cuda_kernel_handle k,
  CUfunction cufunc,
  dim3 gridDim_,
  dim3 blockDim_,
  size_t sharedMem_,
  int arg_cnt,
  const void** args);

/* Convert CUDA kernel address to CUfunction because we may use them from a
 * shared library where this would be invalid in the runtime API. */
static inline void stf_cuda_kernel_add_desc(
  stf_cuda_kernel_handle k,
  const void* func,
  dim3 gridDim_,
  dim3 blockDim_,
  size_t sharedMem_,
  int arg_cnt,
  const void** args)
{
  CUfunction cufunc;
  [[maybe_unused]] cudaError_t res = cudaGetFuncBySymbol(&cufunc, func);
  assert(res == cudaSuccess);

  stf_cuda_kernel_add_desc_cufunc(k, cufunc, gridDim_, blockDim_, sharedMem_, arg_cnt, args);
}

void* stf_cuda_kernel_get_arg(stf_cuda_kernel_handle k, int index);
void stf_cuda_kernel_end(stf_cuda_kernel_handle k);
void stf_cuda_kernel_destroy(stf_cuda_kernel_handle t);

#ifdef __cplusplus
}
#endif
