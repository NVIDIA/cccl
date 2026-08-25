/*===---- HostJIT module teardown runtime ----------------------------------===
 *
 * This is not a header the compiled program includes: it is the source of one
 * object that libnvcc compiles and adds to every library it links, the way a C
 * runtime contributes its own startup and teardown object.  It supplies what a
 * freestanding image is missing around the CUDA module's lifetime:
 *
 *   * atexit(), which the CUDA registration constructor calls to schedule
 *     __cuda_module_dtor -> __cudaUnregisterFatBinary.  That call is the reason
 *     it exists; a call from the compiled program is a compile error instead,
 *     made so by the header the host translation units force-include;
 *   * the hook the OS runs when the library is unloaded, which makes that call:
 *     an entry in .fini_array that the dynamic loader calls from dlclose, the
 *     DLL entry point on Windows;
 *   * the two entry points a loader uses when it wants the teardown to happen
 *     before the unmap rather than during it.
 *
 * Being one object per library is what makes the definitions strong.  The
 * registration constructor's atexit() call has to land here, and it is the only
 * thing standing between the fatbin and a dangling registration; a weak
 * definition would yield to any other definition of the name that reaches the
 * link, and the library would then unregister nothing at unload without saying
 * so.  Strong and once means such a link fails on a duplicate symbol instead.
 *===----------------------------------------------------------------------===*/
#ifndef __HOSTJIT_MODULE_RUNTIME_H__
#define __HOSTJIT_MODULE_RUNTIME_H__

extern "C" {
// The only runtime entry point this object calls, declared rather than included:
// pulling in the CUDA headers would mean compiling the whole host preamble a second
// time for one declaration. cudaError_t is an unscoped enum with small values, so it
// comes back in the register an int does on both platforms, and nothing here reads
// the value -- it is handed to the loader as it is.
int cudaDeviceSynchronize(void);

#if defined(_MSC_VER)
typedef void(__cdecl* __hostjit_atexit_fn)(void);
#else
typedef void (*__hostjit_atexit_fn)(void);
#endif

// What was registered, which is one callback: the module destructor the CUDA
// registration constructor schedules. The compiled program has no way to add to
// it -- the header the host translation units force-include turns a call to
// atexit in the program into a compile error -- so this is a pointer and not a
// table. It is internal to this object, so it is not a symbol at all; what the
// image shows the outside world is the two functions below.
static __hostjit_atexit_fn __hostjit_module_dtor = 0;

// Which of the three states the teardown is in, so that only one caller ever runs
// it: not started, running, done.
enum
{
  __hostjit_fini_idle    = 0,
  __hostjit_fini_running = 1,
  __hostjit_fini_done    = 2
};

static int __hostjit_fini_state = __hostjit_fini_idle;

// On ELF the definition is hidden so that the registration object's call binds
// to it while linking: a hidden symbol is not emitted into .dynsym, so it is
// never looked up at run time and no atexit elsewhere in the process can take
// the call and defer the module destructor back to process exit.
#if defined(_WIN32)
#  define __HOSTJIT_LOCAL_ATTR
#  define __HOSTJIT_EXPORT_ATTR __declspec(dllexport)
#else
#  define __HOSTJIT_LOCAL_ATTR  __attribute__((visibility("hidden")))
// The two functions a loader calls are protected rather than hidden: callable
// from outside, still bound locally inside, which is what keeps the
// .fini_array entry pointing at this image's own copy. Were they
// interposable, a definition of the name already in the process's lookup scope
// would take that call, and the image would tear down someone else's module
// and leave its own fatbin registered.
#  define __HOSTJIT_EXPORT_ATTR __attribute__((visibility("protected")))
#endif

// The registration constructor schedules the module destructor through this, and that is the
// one call it is here for: the compiled program's own would be a compile error, and the
// standard behaviour of running a callback at process exit is not available to a library that
// has no C runtime to keep the list. So it takes one registration and refuses the rest by
// returning non-zero, the way atexit reports failure -- including anything arriving after the
// teardown, since the pointer stays taken and there would be nobody left to make the call.
__HOSTJIT_LOCAL_ATTR int atexit(__hostjit_atexit_fn func)
{
  if (!func)
  {
    return 1;
  }
  __hostjit_atexit_fn taken = 0;
  if (!__atomic_compare_exchange_n(&__hostjit_module_dtor, &taken, func, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
  {
    return 1;
  }
  return 0;
}

// Tears the module down by calling what was registered, which is the module destructor and so
// is what unregisters the fatbin. The OS hooks below call this, and so can a loader that wants
// it done before the unmap rather than during it.
//
// One caller makes the call and the others wait for it, rather than returning: a waiter's own
// caller is about to unmap the image the destructor runs in. The wait is not reentrant, but
// nothing the image runs can reach it -- the only callback is the generated one, which
// unregisters and returns.
__HOSTJIT_EXPORT_ATTR void __cudacc_module_fini(void)
{
  int state = __hostjit_fini_idle;
  if (!__atomic_compare_exchange_n(
        &__hostjit_fini_state, &state, __hostjit_fini_running, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
  {
    while (__atomic_load_n(&__hostjit_fini_state, __ATOMIC_ACQUIRE) != __hostjit_fini_done)
    {
    }
    return;
  }

  // The acquire load pairs with the release in atexit, so the registration the constructor
  // made is visible here even if the teardown runs on another thread. A null means the
  // constructor did not run, which is a module with no fatbin to unregister.
  __hostjit_atexit_fn func = __atomic_load_n(&__hostjit_module_dtor, __ATOMIC_ACQUIRE);
  if (func)
  {
    func();
  }

  __atomic_store_n(&__hostjit_fini_state, __hostjit_fini_done, __ATOMIC_RELEASE);
}

// Waits for the module's GPU work, and, called once more after the fini above,
// pushes out the module unload the runtime has queued. Both belong to the
// loader that unmaps the image, but the call has to land in the runtime this
// image registered its fatbin with: a process can hold more than one, each
// with its own module registry and its own pending-unload queue.
__HOSTJIT_EXPORT_ATTR int __cudacc_module_sync(void)
{
  return cudaDeviceSynchronize();
}

#if !defined(_WIN32)
// ld.so runs .fini_array on dlclose, and the linker builds the array and its
// dynamic tags without any help from a C runtime, so this works in a
// freestanding image.
__attribute__((used,
               retain,
               section(".fini_array"),
               aligned(__alignof__(__hostjit_atexit_fn)))) static __hostjit_atexit_fn __hostjit_module_fini_entry =
  __cudacc_module_fini;
#else
// Windows has no equivalent of .fini_array: what the OS runs on load and unload is the DLL's
// entry point, and a DLL built without a C runtime has none (it used to be linked /NOENTRY).
// So this object provides one, named on the link line with /ENTRY, doing both halves of what
// the CRT would do: run the static constructors on attach, which is where the fatbin gets
// registered, and call the module finalizer on detach.
//
// The constructors are found the way the MSVC CRT finds them, through markers that sort
// around the compiler's own .CRT$XCU contributions, each pragma declaring the section its
// marker goes into. None of this needs Windows headers, which the freestanding compilation
// does not have, so the one API the entry point calls is declared by hand.
typedef void(__cdecl* __hostjit_init_fn)(void);

#  pragma section(".CRT$XCA", long, read)
__declspec(allocate(".CRT$XCA")) __hostjit_init_fn __hostjit_ctors_begin[] = {0};
#  pragma section(".CRT$XCZ", long, read)
__declspec(allocate(".CRT$XCZ")) __hostjit_init_fn __hostjit_ctors_end[]   = {0};

__declspec(dllimport) int __stdcall DisableThreadLibraryCalls(void* module);

int __stdcall __hostjit_dll_entry(void* instance, unsigned long reason, void* reserved)
{
  enum
  {
    __hostjit_process_detach = 0,
    __hostjit_process_attach = 1
  };
  if (reason == __hostjit_process_attach)
  {
    // Nothing in the image is per-thread, so ask not to be called again for
    // every thread the host process creates or exits.
    DisableThreadLibraryCalls(instance);
    for (__hostjit_init_fn* it = __hostjit_ctors_begin; it < __hostjit_ctors_end; ++it)
    {
      if (*it)
      {
        (*it)();
      }
    }
  }
  // A non-null `reserved` on detach means the process is exiting rather than
  // the library being unloaded. Nothing is worth unregistering then, and the
  // runtime may already be torn down, so leave it alone.
  else if (reason == __hostjit_process_detach && reserved == 0)
  {
    __cudacc_module_fini();
  }
  return 1; // a zero return from the entry point fails LoadLibrary
}
#endif
} // extern "C"

#endif // __HOSTJIT_MODULE_RUNTIME_H__
