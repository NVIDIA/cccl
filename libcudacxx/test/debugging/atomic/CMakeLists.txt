libcudacxx_add_pretty_printer_test(
  NAME atomic
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_integer atomic.integer value
    --case inspect_boolean atomic.boolean value
    --case inspect_small_object atomic.small_object value
    --case inspect_floating_point atomic.floating_point value
    --case inspect_pointer atomic.pointer value
    --case inspect_locked atomic.locked value
    --case inspect_alias atomic.alias value
    --case inspect_nested atomic.nested values
    --case inspect_reference atomic.reference value
    --case inspect_locked_reference atomic.locked_reference value
    --case inspect_cuda_integer atomic.cuda_integer value
    --case inspect_cuda_reference atomic.cuda_reference value
    --case inspect_cuda_device_integer atomic.cuda_device_integer value
    --case inspect_cuda_block_reference atomic.cuda_block_reference value
    --case inspect_cuda_thread_integer atomic.cuda_thread_integer value
    --case inspect_before_update atomic.update.before value
    --case inspect_after_update atomic.update.after value
    --case inspect_reference_before_update atomic.reference_update.before value
    --case inspect_reference_after_update atomic.reference_update.after value
    # gersemi: on
)
