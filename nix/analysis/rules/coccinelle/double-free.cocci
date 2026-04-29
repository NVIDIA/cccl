// Detect potential double-free patterns.
// Finds cases where a pointer is freed twice without reassignment.
@@
expression ptr;
expression E;
@@

  free(ptr);
  ... when != ptr = E
*  free(ptr);
