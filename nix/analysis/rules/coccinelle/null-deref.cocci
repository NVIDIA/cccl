// Detect potential null pointer dereferences after allocation.
// Finds cases where a pointer from malloc/calloc/realloc is used
// without a NULL check.
@@
expression E;
expression ptr;
@@

  ptr = \(malloc\|calloc\|realloc\)(...);
  ... when != if (ptr == NULL) { ... }
      when != if (!ptr) { ... }
      when != if (ptr != NULL) { ... }
      when != if (ptr) { ... }
*  ptr->E
