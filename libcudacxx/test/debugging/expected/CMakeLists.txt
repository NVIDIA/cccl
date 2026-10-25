libcudacxx_add_pretty_printer_test(
  NAME expected
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_value_int expected.value_int values
    --case inspect_value_double expected.value_double values
    --case inspect_value_zero expected.value_zero values
    --case inspect_error_int expected.error_int values
    --case inspect_error_struct expected.error_struct values
    --case inspect_before_update expected.update.before values
    --case inspect_after_update expected.update.after values
    --case inspect_alias expected.alias values
    --case inspect_nested expected.nested values
    --case inspect_void_value expected_void.value values
    --case inspect_void_error expected_void.error values
    # gersemi: on
)
