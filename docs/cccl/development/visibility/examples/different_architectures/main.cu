#include "kernel.cuh"

void non_inlined_function();

int main()
{
  some_class_with_kernel with_inline{1};
  printf("a: value of class with inlined constructor: %d\n", with_inline.val_);

  non_inlined_function();
}
