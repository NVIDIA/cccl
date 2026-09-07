libcudacxx_add_pretty_printer_test(
  NAME complex
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_std_default complex.std.default value
    --case inspect_std_double complex.std.double value
    --case inspect_cuda_float complex.cuda.float value
    --case inspect_alias complex.alias value
    --case inspect_nested complex.nested values
    --case inspect_before_update complex.update.before value
    --case inspect_after_update complex.update.after value
    # gersemi: on
)
