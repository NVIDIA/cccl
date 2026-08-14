libcudacxx_add_pretty_printer_test(
  NAME tuple
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_basic tuple.basic values
    --case inspect_empty tuple.empty values
    --case inspect_single tuple.single values
    --case inspect_ebco tuple.ebco values
    --case inspect_ebco_first tuple.ebco_first values
    --case inspect_ebco_last tuple.ebco_last values
    --case inspect_ebco_adjacent tuple.ebco_adjacent values
    --case inspect_pointer tuple.pointer values
    --case inspect_nested_empty tuple.nested_empty values
    --case inspect_nested tuple.nested values
    --case inspect_reference tuple.reference values
    --case inspect_const tuple.const values
    --case inspect_vector tuple.vector.0 "values[0]"
    --case inspect_vector tuple.vector.1 "values[1]"
    --case inspect_before_update tuple.update.before values
    --case inspect_after_update tuple.update.after values
    # gersemi: on
)
