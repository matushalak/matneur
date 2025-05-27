# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="adex_cython",
        sources=["adex_cython.pyx"],
        include_dirs=[numpy.get_include()],   # <— here’s the magic
    )
]

setup(
    name="adex_cython",
    ext_modules=cythonize(extensions, language_level="3"),
    zip_safe=False,
)


exts = [
    Extension(
        "adex_rk4",
        ["adex_rk4.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(name="adex_rk4",
      ext_modules=cythonize(exts, language_level="3"))