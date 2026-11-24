CodeRabbit
==========

This page explains how to configure and use CodeRabbit for CCCL pull request
review. For the complete product documentation, see the
`CodeRabbit documentation <https://docs.coderabbit.ai/>`__.

Configuration
-------------

CCCL configures CodeRabbit through ``.coderabbit.yaml`` in the repository root.
The configuration in the pull request branch is used for that review.

When setting up or updating CodeRabbit:

#. Keep repository-specific settings in ``.coderabbit.yaml``.
#. Use ``@coderabbitai configuration`` on a pull request to inspect the
   resolved configuration. This is useful when checking whether CodeRabbit is
   using the expected repository settings for that pull request.
#. Use ``@coderabbitai generate configuration`` to export the resolved
   configuration if a new baseline is needed. This is useful when moving
   settings into a reviewable repository configuration file.
#. Keep configuration changes small and reviewable.
#. Use ``reviews.path_instructions`` for path-specific review guidance.
#. Use ``knowledge_base.code_guidelines.filePatterns`` for CCCL guidance files
   that CodeRabbit should read as review context.

The CCCL configuration should keep comments focused on correctness, API
stability, performance, security, and other high-impact issues. Avoid enabling
features that add noisy generated comments or code by default.

Pull Request Reviews
--------------------

Automatic reviews may be disabled or restricted by the repository
configuration. Maintainers can always request review explicitly from a pull
request comment:

.. code-block:: text

   @coderabbitai review

Use a full review when the pull request should be reviewed again from scratch:

.. code-block:: text

   @coderabbitai full review

Other useful commands:

- ``@coderabbitai help`` shows the current command reference.
- ``@coderabbitai configuration`` shows the active configuration.
- ``@coderabbitai summary`` can be placed in the pull request description as a
  placeholder for the generated summary.
- ``@coderabbitai pause`` and ``@coderabbitai resume`` pause or resume
  automatic review behavior. This is useful when a pull request is still being
  updated frequently and should not be reviewed again until it is ready.
- ``@coderabbitai ignore`` can be placed in the pull request description to
  disable automatic reviews.

Review Guidance
---------------

Treat CodeRabbit feedback as review assistance, not as a merge requirement by
itself. Maintainers remain responsible for deciding whether comments are
actionable and whether a pull request has adequate tests and CI coverage.
