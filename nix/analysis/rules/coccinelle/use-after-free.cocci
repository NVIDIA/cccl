// Detect potential use-after-free patterns.
// Finds cases where a pointer is used after being freed.
@@
expression ptr;
expression E;
@@

  free(ptr);
  ... when != ptr = E
*  ptr
