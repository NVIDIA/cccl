libcudacxx_add_pretty_printer_test(
  NAME array
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_normal 1 array.normal normal
    --case inspect_empty 1 array.empty empty
    --case inspect_nested 1 array.nested nested
    --case inspect_alias 1 array.alias alias
    --case inspect_before_update 1 array.update.before updated_values
    --case inspect_after_update 1 array.update.after updated_values
    # gersemi: on
)
