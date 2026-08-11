libcudacxx_add_pretty_printer_test(
  NAME memory_pool_managed
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_no_init memory_pool_managed.no_init value
    --case inspect_shared_no_init memory_pool_managed.shared_no_init value
    # gersemi: on
)
