.. _cccl-development-module-debugger-setup:

=================
General Debugging
=================

Debugger Pretty Printers
========================

libcudacxx ships custom pretty printers for its types under
``libcudacxx/share/libcudacxx``. They render CCCL types in a readable form and, for
device-accessible data, copy the contents back to the host so the elements can be
inspected. Two independent implementations are provided:

- ``libcudacxx/share/libcudacxx/gdb`` - printers for GDB.
- ``libcudacxx/share/libcudacxx/lldb`` - printers for LLDB.

Each directory has an ``__init__.py`` entry point that registers every printer. The
repository root contains a ``.gdbinit`` and a ``.lldbinit`` that load the matching entry
point for you, so the simplest way to enable the printers is to let the debugger pick up
these files.

.. important::

   ``lldb`` and ``gdb`` only inspect the following locations when looking for init
   dotfiles (in the given order):

   #. Home directory config files (usually ``~/.lldbinit`` or ``~/.config/gdb/gdbinit`` on
      Linux, but check the respective manuals for specifics).
   #. The current working directory.

   They do **not** walk up the directory stack like most tools. So if you have a
   ``.lldbinit`` in the parent directory, ``lldb`` will **not** load it. For this reason,
   you **must** run the debugger from the root CCCL directory in order for automatic
   loading of the pretty printers to work.


In addition to not loading parent directory dot-files, ``gdb`` or ``lldb`` will load not
dotfiles unless you explicitly allow them. The following sections explain how to enable
this for each debugger.

.. note::

   The following is **not** needed when working inside a devcontainer. devcontainers
   already have the following set up.

   If they don't, and automatic loading of the pretty printers does not work, then this is
   a bug and should be fixed.

   It is only needed for bare metal builds.

GDB
---

By default GDB does not source a ``.gdbinit`` from the current directory, and it guards
auto-loaded scripts with the ``auto-load safe-path`` setting. Add the repository root to
your ``~/.gdbinit`` (or ``~/.config/gdb/gdbinit`` if you have ``XDG_CONFIG_HOME`` set) so
the project's ``.gdbinit`` is trusted and loaded::

  add-auto-load-safe-path /absolute/path/to/cccl
  set auto-load local-gdbinit on

Launch GDB from the repository root and the printers should register automatically.

Verify that the printers are active with ``info pretty-printer``.

To load the printers without depending on the working directory - for example
from a global ``~/.gdbinit`` - ``source`` the entry point by absolute path
instead::

  source /absolute/path/to/cccl/libcudacxx/share/libcudacxx/gdb/__init__.py

``source`` runs the script directly and is not subject to the ``auto-load safe-path``
restriction.

LLDB
----

LLDB only reads ``.lldbinit`` from your home directory unless you opt in to loading one
from the current working directory. Enable that once in your ``~/.lldbinit`` (``lldb``
seemingly does not respect ``XDG_CONFIG_HOME``)::

  settings set target.load-cwd-lldbinit true

This is a trust decision, since the local file runs arbitrary Python. Launch LLDB from the
repository root and the project's ``.lldbinit`` imports the formatters automatically.

To load the formatters without depending on the working directory, add the absolute path
to your ``~/.lldbinit`` instead::

  command script import "/absolute/path/to/cccl/libcudacxx/share/libcudacxx/lldb/__init__.py"

The entry point's ``__lldb_init_module`` hook defines and enables an LLDB type category
for the formatters. Print any CCCL value with the usual commands (``v``, ``frame
variable``, or ``dwim-print``).
