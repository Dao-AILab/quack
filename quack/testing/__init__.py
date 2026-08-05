# Copyright (c) 2026, Tri Dao.
"""Reusable pytest plugin for QuACK test workflows.

This subpackage's only contents is the reusable pytest plugin in
:mod:`quack.testing.pytest_plugin`, which wires the ``--async-compile`` pool
(defer-and-retry kernel compilation) into a pytest run::

    # In a downstream project's conftest.py:
    pytest_plugins = ["quack.testing.pytest_plugin"]
"""
