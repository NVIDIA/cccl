libcudacxx_add_pretty_printer_test(
  NAME buffer
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_normal 1 buffer.normal normal_values
    --case inspect_alias 1 buffer.alias aliased_values
    --case inspect_vector 1 buffer.vector.0 "buffer_vector[0]"
    --case inspect_vector 1 buffer.vector.1 "buffer_vector[1]"
    --case inspect_host_device 1 buffer.host_device host_device_values
    --case inspect_empty 1 buffer.empty empty_values
    --case inspect_before_update 1 buffer.update.before updated_values
    --case inspect_after_update 1 buffer.update.after updated_values
    # gersemi: on
)
