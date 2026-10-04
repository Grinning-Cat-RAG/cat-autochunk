"""pytest configuration of the plugin.

The folder of the plugin may not be a valid Python package name (e.g. a "cat-autochunk" checkout): when the tests run
under pytest, the folder is registered as the package "autochunk_plugin", so that the tests can import the modules of
the plugin (and their relative imports work). Inside the Cat, the tests are part of the package of the plugin and use
relative imports.

NOTE: the Cat imports every .py file of a plugin (and scans it for forbidden constructs): this file and the tests must
not import pytest at module level, nor use dynamic imports.
"""
import os
import sys
import types

ALIAS = "autochunk_plugin"

if "pytest" in sys.modules and ALIAS not in sys.modules:
    _package = types.ModuleType(ALIAS)
    setattr(_package, "__path__", [os.path.dirname(os.path.abspath(__file__))])
    sys.modules[ALIAS] = _package
