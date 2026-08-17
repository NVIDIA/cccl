libcudacxx_add_pretty_printer_test(
  NAME optional
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_disengaged optional.disengaged values
    --case inspect_engaged_int optional.engaged_int values
    --case inspect_engaged_double optional.engaged_double values
    # gersemi: on
)
