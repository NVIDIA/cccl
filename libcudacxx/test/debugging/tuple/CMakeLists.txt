libcudacxx_add_pretty_printer_test(
  NAME tuple
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_basic 1 tuple.basic basic_values
    --case inspect_empty 1 tuple.empty empty_values
    --case inspect_single 1 tuple.single single_values
    --case inspect_ebco 1 tuple.ebco ebco_values
    --case inspect_nested 1 tuple.nested nested_values
    --case inspect_reference 1 tuple.reference reference_values
    --case inspect_const 1 tuple.const const_values
    --case inspect_vector 1 tuple.vector.0 "tuple_vector[0]"
    --case inspect_vector 1 tuple.vector.1 "tuple_vector[1]"
    --case inspect_before_update 1 tuple.update.before mutable_values
    --case inspect_after_update 1 tuple.update.after mutable_values
    # gersemi: on
)
