.. _contributors-vscode-cmake-tools:

Using CMake Presets via VS Code GUI extension
================================================

Recommended when using DevContainers.

The recommended way to use CMake Presets is via the VS Code extension `CMake Tools
<https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools>`_, already included in
`CCCL's DevContainers <https://github.com/NVIDIA/cccl/blob/main/.devcontainer/README.md>`_. As soon as
you install the extension you would be able to see the sidebar menu below.

.. image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/cmaketools_sidebar.png
   :alt: cmaketools sidebar

You can specify the desired CMake Preset by clicking the "Select Configure Preset" button under the
"Configure" node (see image below).

.. image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/cmaketools_presets.png
   :alt: cmaketools presets

After that you can select the default build target from the "Build" node. As soon as you expand it, a
list will appear with all the available targets that are included within the preset you selected. For
example if you had selected the ``all-dev`` preset VS Code will display all the available targets we
have in cccl.

.. image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/cmaketools_targets.png
   :alt: cmaketools targets

You can build the selected target by pressing the gear button
(|gear|) at the bottom of the VS Code window.

Alternatively you can select the desired target from either the "Debug" or "Launch" drop down menu
(for debugging or running correspondingly). In that case after you select the target and either press
"Run" (|run|) or "Debug" (|debug|) the target will build on its own before running without the user
having to build it explicitly from the gear button.

.. |gear| image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/build_button.png
.. |run| image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/run.png
.. |debug| image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/debug.png

----

We encourage users who want to debug device code to install the `Nsight Visual Studio Code Edition
extension <https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition>`_ that
enables the VS Code frontend for ``cuda-gdb``. To use it you should launch from the sidebar menu instead
of pressing the "Debug" button from the bottom menu.

.. image:: https://raw.githubusercontent.com/NVIDIA/cccl/main/.devcontainer/img/nsight.png
   :alt: nsight
