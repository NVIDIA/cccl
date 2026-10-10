libcudacxx_add_pretty_printer_test(
  NAME annotated_ptr
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_streaming annotated_ptr.streaming ptr
    --case inspect_persisting annotated_ptr.persisting ptr
    --case inspect_global annotated_ptr.global ptr
    --case inspect_null annotated_ptr.null ptr
    # gersemi: on
)
