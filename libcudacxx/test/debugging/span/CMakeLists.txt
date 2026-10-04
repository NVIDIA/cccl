libcudacxx_add_pretty_printer_test(
  NAME span
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_static span.static values
    --case inspect_dynamic span.dynamic values
    --case inspect_static_empty span.static_empty values
    --case inspect_default span.default values
    --case inspect_const span.const values
    --case inspect_nested span.nested values
    --case inspect_alias span.alias values
    --case inspect_before_update span.update.before values
    --case inspect_after_update span.update.after values
    # gersemi: on
)
