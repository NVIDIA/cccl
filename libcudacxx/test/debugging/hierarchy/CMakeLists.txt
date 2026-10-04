libcudacxx_add_pretty_printer_test(
  NAME hierarchy
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_static hierarchy.static values
    --case inspect_dynamic hierarchy.dynamic values
    --case inspect_partial hierarchy.partial values
    --case inspect_derived hierarchy.derived values
    --case inspect_vector hierarchy.vector.0 "values[0]"
    --case inspect_vector hierarchy.vector.1 "values[1]"
    --case inspect_before_update hierarchy.update.before values
    --case inspect_after_update hierarchy.update.after values
    # gersemi: on
)
