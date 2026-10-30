libcudacxx_add_pretty_printer_test(
  NAME array
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_normal array.normal values
    --case inspect_empty array.empty values
    --case inspect_nested array.nested values
    --case inspect_alias array.alias values
    --case inspect_before_update array.update.before values
    --case inspect_after_update array.update.after values
    # gersemi: on
)
