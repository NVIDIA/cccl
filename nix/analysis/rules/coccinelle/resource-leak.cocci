// Detect potential resource leaks from fopen without fclose.
// Finds FILE pointers that may not be closed on all paths.
@@
expression fp;
expression E;
@@

  fp = fopen(...);
  ... when != fclose(fp)
      when != fp = E
*  return ...;
