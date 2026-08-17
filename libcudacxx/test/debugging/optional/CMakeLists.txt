libcudacxx_add_pretty_printer_test(
  NAME optional
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_disengaged optional.disengaged values
    --case inspect_engaged_int optional.engaged_int values
    --case inspect_engaged_double optional.engaged_double values
    --case inspect_before_update optional.update.before values
    --case inspect_after_update optional.update.after values
    --case inspect_alias optional.alias values
    --case inspect_reference optional.reference values
    --case inspect_nested optional.nested values
    # gersemi: on
)
