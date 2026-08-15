libcudacxx_add_pretty_printer_test(
  NAME memory_pool_managed
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_owning memory_pool_managed.owning value
    --case inspect_ref memory_pool_managed.ref value
    --case inspect_no_init memory_pool_managed.no_init value
    --case inspect_shared memory_pool_managed.shared value
    --case inspect_shared_no_init memory_pool_managed.shared_no_init value
    --case inspect_in_use memory_pool_managed.in_use value
    --case inspect_shared_in_use memory_pool_managed.shared_in_use value
    # gersemi: on
)
