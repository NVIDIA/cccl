libcudacxx_add_pretty_printer_test(
  NAME inplace_vector
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_empty 1 inplace_vector.empty empty
    --case inspect_partial 1 inplace_vector.partial partial
    --case inspect_full 1 inplace_vector.full full
    --case inspect_zero_capacity 1 inplace_vector.zero_capacity zero
    --case inspect_nested 1 inplace_vector.nested nested
    --case inspect_before_update 1 inplace_vector.update.before updated
    --case inspect_after_update 1 inplace_vector.update.after updated
    # gersemi: on
)
