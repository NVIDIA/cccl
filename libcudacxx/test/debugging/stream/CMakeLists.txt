libcudacxx_add_pretty_printer_test(
  NAME stream
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_without_context stream.without_context value
    --case inspect_owning stream.owning value
    --case inspect_ref stream.ref value
    --case inspect_alias stream.alias value
    --case inspect_default stream.default value
    --case inspect_legacy stream.legacy value
    --case inspect_per_thread stream.per_thread value
    --case inspect_invalid stream.invalid value
    --case inspect_moved_from stream.moved_from value
    --case inspect_summary stream.summary values
    --case inspect_before_update stream.update.before value
    --case inspect_after_update stream.update.after value
    --case inspect_capturing stream.capturing value
    --case inspect_after_capture stream.after_capture value
    # gersemi: on
)

target_link_libraries(
  libcudacxx.test.debugging.stream
  PRIVATE CUDA::cuda_driver
)
