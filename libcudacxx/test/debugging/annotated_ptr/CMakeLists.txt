libcudacxx_add_pretty_printer_test(
  NAME annotated_ptr
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_streaming 1 annotated_ptr.streaming streaming_ptr
    --case inspect_persisting 1 annotated_ptr.persisting persisting_ptr
    --case inspect_global 1 annotated_ptr.global global_ptr
    --case inspect_null 1 annotated_ptr.null null_ptr
    # gersemi: on
)
