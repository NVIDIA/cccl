libcudacxx_add_pretty_printer_test(
  NAME memory_resource
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_device memory_resource.device resource
    --case inspect_host_device memory_resource.host_device resource
    --case inspect_alias memory_resource.alias resource
    # gersemi: on
)
