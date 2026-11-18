"""Build optional Cython extensions for boolmore."""

from __future__ import annotations

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:
    ext_modules = []
else:
    ext_modules = cythonize(
        [
            Extension(
                "boolmore._get_uni_rr",
                sources=["src/boolmore/_get_uni_rr.pyx"],
            ),
            Extension(
                "boolmore._ldoi_closure",
                sources=["src/boolmore/_ldoi_closure.pyx"],
            ),
        ],
        language_level=3,
    )

setup(ext_modules=ext_modules)
