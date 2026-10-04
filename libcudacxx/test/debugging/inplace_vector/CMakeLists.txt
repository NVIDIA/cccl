libcudacxx_add_pretty_printer_test(
  NAME inplace_vector
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_empty inplace_vector.empty values
    --case inspect_partial inplace_vector.partial values
    --case inspect_full inplace_vector.full values
    --case inspect_zero_capacity inplace_vector.zero_capacity values
    --case inspect_nested inplace_vector.nested values
    --case inspect_before_update inplace_vector.update.before values
    --case inspect_after_update inplace_vector.update.after values
    # gersemi: on
)
