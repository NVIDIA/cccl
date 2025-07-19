// TODO use CCCL_C_EXTERN_C_BEGIN/CCCL_C_EXTERN_C_END
#ifdef __cplusplus
extern "C" {
#endif

typedef struct stf_ctx_handle_t* stf_ctx_handle;

void stf_ctx_create(stf_ctx_handle* ctx);
void stf_ctx_finalize(stf_ctx_handle ctx);

typedef struct stf_logical_data_handle_t* stf_logical_data_handle;

void stf_logical_data(stf_ctx_handle ctx, stf_logical_data_handle *ld, void *addr, size_t sz);
void stf_logical_data_destroy(stf_ctx_handle ctx, stf_logical_data_handle ld);

typedef struct stf_task_handle_t* stf_task_handle;

#ifdef __cplusplus
}
#endif
