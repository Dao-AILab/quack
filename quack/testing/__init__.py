# Copyright (c) 2026, Tri Dao.
"""Reusable pytest plugin for QuACK test workflows.

This subpackage's only contents is the reusable pytest plugin in
:mod:`quack.testing.pytest_plugin`, which wires the ``--compile-only`` CLI
flag and the FakeTensorMode lifecycle into a pytest run::

    # In a downstream project's conftest.py:
    pytest_plugins = ["quack.testing.pytest_plugin"]
"""
