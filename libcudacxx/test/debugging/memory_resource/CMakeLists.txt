libcudacxx_add_pretty_printer_test(
  NAME memory_resource
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_device 1 memory_resource.device device_resource
    --case inspect_host_device 1 memory_resource.host_device host_device_resource
    --case inspect_alias 1 memory_resource.alias aliased_resource
    # gersemi: on
)
