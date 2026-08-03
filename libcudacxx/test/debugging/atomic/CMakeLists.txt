libcudacxx_add_pretty_printer_test(
  NAME atomic
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_integer 1 atomic.integer integer
    --case inspect_boolean 1 atomic.boolean boolean
    --case inspect_small_object 1 atomic.small_object small_object
    --case inspect_floating_point 1 atomic.floating_point floating_point
    --case inspect_pointer 1 atomic.pointer pointer
    --case inspect_locked 1 atomic.locked locked
    --case inspect_alias 1 atomic.alias alias
    --case inspect_nested 1 atomic.nested nested
    --case inspect_reference 1 atomic.reference reference
    --case inspect_locked_reference 1 atomic.locked_reference locked_reference
    --case inspect_cuda_integer 1 atomic.cuda_integer cuda_integer
    --case inspect_cuda_reference 1 atomic.cuda_reference cuda_reference
    --case inspect_cuda_device_integer 1 atomic.cuda_device_integer cuda_device_integer
    --case inspect_cuda_block_reference 1 atomic.cuda_block_reference cuda_block_reference
    --case inspect_cuda_thread_integer 1 atomic.cuda_thread_integer cuda_thread_integer
    --case inspect_before_update 1 atomic.update.before updated
    --case inspect_after_update 1 atomic.update.after updated
    --case inspect_reference_before_update 1 atomic.reference_update.before updated_reference
    --case inspect_reference_after_update 1 atomic.reference_update.after updated_reference
    # gersemi: on
)
