libcudacxx_add_pretty_printer_test(
  NAME memory_resource
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_in_situ memory_resource.in_situ resource
    --case inspect_heap memory_resource.heap resource
    --case inspect_synchronous memory_resource.synchronous resource
    --case inspect_host_device memory_resource.host_device resource
    --case inspect_propertyless memory_resource.propertyless resource
    --case inspect_alias memory_resource.alias resource
    --case inspect_empty memory_resource.empty resource
    --case inspect_before_reset memory_resource.reset.before resource
    --case inspect_after_reset memory_resource.reset.after resource
    --case inspect_before_move memory_resource.move.before resource
    --case inspect_after_move_source memory_resource.move.after.source resource
    --case inspect_after_move_target memory_resource.move.after.target resource
    # gersemi: on
)
