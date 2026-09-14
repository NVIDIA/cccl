libcudacxx_add_pretty_printer_test(
  NAME expected
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_value expected.value result
    --case inspect_error expected.error result
    --case inspect_void_value expected.void_value result
    --case inspect_void_error expected.void_error result
    --case inspect_before_update expected.update.before result
    --case inspect_after_update expected.update.after result
    # gersemi: on
)
