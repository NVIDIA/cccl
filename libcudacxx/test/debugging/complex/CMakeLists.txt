libcudacxx_add_pretty_printer_test(
  NAME complex
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_std_default 1 complex.std.default std_default
    --case inspect_std_double 1 complex.std.double std_double
    --case inspect_cuda_float 1 complex.cuda.float cuda_float
    --case inspect_alias 1 complex.alias alias
    --case inspect_nested 1 complex.nested nested
    --case inspect_before_update 1 complex.update.before updated
    --case inspect_after_update 1 complex.update.after updated
    # gersemi: on
)
