libcudacxx_add_pretty_printer_test(
  NAME memory_pool_pinned
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_owning memory_pool_pinned.owning value
    --case inspect_ref memory_pool_pinned.ref value
    --case inspect_shared memory_pool_pinned.shared value
    --case inspect_in_use memory_pool_pinned.in_use value
    --case inspect_shared_in_use memory_pool_pinned.shared_in_use value
    # gersemi: on
)
