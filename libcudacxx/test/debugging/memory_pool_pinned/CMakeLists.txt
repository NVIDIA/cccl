libcudacxx_add_pretty_printer_test(
  NAME memory_pool_pinned
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_owning memory_pool_pinned.owning value
    --case inspect_shared memory_pool_pinned.shared value
    # gersemi: on
)
