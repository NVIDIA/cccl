target_compile_options(
  ${example_target}
  PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>: --extended-lambda>
)
