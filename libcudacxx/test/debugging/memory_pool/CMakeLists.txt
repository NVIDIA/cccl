libcudacxx_add_pretty_printer_test(
  NAME memory_pool
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_owning memory_pool.owning value
    --case inspect_ref memory_pool.ref value
    --case inspect_alias memory_pool.alias value
    --case inspect_no_init memory_pool.no_init value
    --case inspect_shared memory_pool.shared value
    --case inspect_shared_copy memory_pool.shared_copy value
    --case inspect_shared_no_init memory_pool.shared_no_init value
    --case inspect_nested memory_pool.nested values
    --case inspect_before_update memory_pool.update.before value
    --case inspect_after_update memory_pool.update.after value
    --case inspect_in_use memory_pool.in_use value
    --case inspect_after_release memory_pool.after_release value
    --case inspect_shared_in_use memory_pool.shared_in_use value
    # gersemi: on
)
