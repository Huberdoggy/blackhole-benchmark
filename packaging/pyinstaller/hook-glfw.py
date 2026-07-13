"""Collect GLFW wheel-provided shared libraries.

The glfw package loads glfw3.dll from its package directory at import time on
Windows. PyInstaller does not always infer that ctypes lookup, so keep the DLLs
beside glfw.library inside the frozen bundle.
"""

from PyInstaller.utils.hooks import collect_dynamic_libs


binaries = collect_dynamic_libs("glfw", destdir="glfw")
