// TODO use CCCL_C_EXTERN_C_BEGIN/CCCL_C_EXTERN_C_END
#ifdef __cplusplus
extern "C" {
#endif

typedef struct stf_ctx_handle stf_ctx_handle;

void stf_ctx_create(stf_ctx_handle* handle);
void stf_ctx_finalize(stf_ctx_handle* handle);

struct stf_task_handle
{
  void* handle;
};

struct stf_logical_data_handle
{
  void* handle;
};

#ifdef __cplusplus
}
#endif
