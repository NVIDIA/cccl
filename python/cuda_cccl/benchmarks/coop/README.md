We benchmark block- and warp-level algorithms with a kernel that does
nothing else. The challenge is keeping the compiler from optimizing
away the work while ensuring the algorithm dominates runtime.

To avoid memory traffic, we generate input data from `clock()`. This
keeps the values unknown at compile time without paying the cost of
global loads. We also avoid constants, which would enable further
optimization and skew results away from realistic workloads.

To make the algorithm dominate execution time, we call it in an
unrolled loop. To prevent the compiler from collapsing identical calls,
each iteration depends on the previous one: the output of one call
becomes the input to the next.

Finally, we introduce a side effect so the compiler must keep the code.
We do this with a write behind a condition that is never true, avoiding
the cost of an actual store while still preventing dead-code removal.
