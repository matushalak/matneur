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