libcudacxx_add_pretty_printer_test(
  NAME optional
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_engaged optional.engaged values
    --case inspect_disengaged optional.disengaged values
    --case inspect_nested optional.nested values
    --case inspect_nested_disengaged optional.nested_disengaged values
    --case inspect_non_trivial optional.non_trivial values
    --case inspect_alias optional.alias values
    --case inspect_reference optional.reference values
    --case inspect_reference_disengaged optional.reference_disengaged values
    --case inspect_before_update optional.update.before values
    --case inspect_after_update optional.update.after values
    # gersemi: on
)
