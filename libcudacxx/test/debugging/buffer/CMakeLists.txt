libcudacxx_add_pretty_printer_test(
  NAME buffer
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_normal buffer.normal values
    --case inspect_alias buffer.alias values
    --case inspect_vector buffer.vector.0 "values[0]"
    --case inspect_vector buffer.vector.1 "values[1]"
    --case inspect_host_device buffer.host_device values
    --case inspect_empty buffer.empty values
    --case inspect_before_update buffer.update.before values
    --case inspect_after_update buffer.update.after values
    # gersemi: on
)
