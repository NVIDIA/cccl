libcudacxx_add_pretty_printer_test(
  NAME shared_resource
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_unique 1 shared_resource.unique unique_resource
    --case inspect_shared 1 shared_resource.shared second_reference
    --case inspect_alias 1 shared_resource.alias aliased_resource
    --case inspect_moved_into 1 shared_resource.moved_into moved_resource
    --case inspect_moved_from 1 shared_resource.moved_from donor_resource
    # gersemi: on
)
