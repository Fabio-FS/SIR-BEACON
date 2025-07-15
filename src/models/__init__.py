"""
BEACON epidemic model collection

This package contains various compartmental epidemic models that capture
different interventions and social structures.

All Python files in this directory will be automatically imported.
"""

import os
import importlib
import pkgutil
import sys
from pathlib import Path

# Get the current directory
_current_dir = Path(__file__).parent

# Automatically import all modules in this package
for (_, module_name, _) in pkgutil.iter_modules([str(_current_dir)]):
    # Skip __init__ itself
    if module_name != "__init__":
        # Import the module
        module = importlib.import_module(f".{module_name}", package=__name__)
        
        # Add all its attributes to the package namespace
        for attribute_name in dir(module):
            # Skip private attributes (starting with _)
            if not attribute_name.startswith('_'):
                globals()[attribute_name] = getattr(module, attribute_name)

# Clean up namespace to avoid exposing implementation details
del os, importlib, pkgutil, sys, Path, _current_dir