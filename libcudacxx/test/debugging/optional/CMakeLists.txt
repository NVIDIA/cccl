libcudacxx_add_pretty_printer_test(
  NAME optional
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_disengaged optional.disengaged values
    --case inspect_engaged_int optional.engaged_int values
    --case inspect_engaged_double optional.engaged_double values
    --case inspect_engaged_zero optional.engaged_zero values
    --case inspect_before_update optional.update.before values
    --case inspect_after_update optional.update.after values
    --case inspect_alias optional.alias values
    --case inspect_nested optional.nested values
    --case inspect_ref_disengaged optional_ref.disengaged values
    --case inspect_ref_engaged optional_ref.engaged values
    --case inspect_ref_engaged_zero optional_ref.engaged_zero values
    --case inspect_ref_before_update optional_ref.update.before values
    --case inspect_ref_after_update optional_ref.update.after values
    --case inspect_ref_alias optional_ref.alias values
    --case inspect_ref_nested optional_ref.nested values
    # gersemi: on
)
