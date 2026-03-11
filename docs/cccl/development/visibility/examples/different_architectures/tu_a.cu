#include "kernel.cuh"

template <class T>
some_class_with_kernel<T>::some_class_with_kernel()
{
  val_ = use_kernel();
}

void non_inlined_function()
{
  some_class_with_kernel with_inline{1};
  printf("a: value of class with inlined constructor: %d\n", with_inline.val_);

  some_class_with_kernel from_library{};
  printf("a: value of class with constructor from library: %d\n", from_library.val_);
}
