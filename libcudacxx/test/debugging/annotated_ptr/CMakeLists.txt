libcudacxx_add_pretty_printer_test(
  NAME annotated_ptr
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_streaming annotated_ptr.streaming streaming_ptr
    --case inspect_persisting annotated_ptr.persisting persisting_ptr
    --case inspect_global annotated_ptr.global global_ptr
    --case inspect_null annotated_ptr.null null_ptr
    # gersemi: on
)
