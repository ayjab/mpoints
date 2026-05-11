import setuptools
from Cython.Build import cythonize
import numpy
import platform

# Run in the console: python setup.py build_ext --inplace

libraries = ["m"] if platform.system() != "Windows" else []

ext_modules = [
    setuptools.Extension(
        name="hybrid_hawkes_exp_cython",
        sources=["hybrid_hawkes_exp_cython.pyx"],
        libraries=libraries,
        extra_compile_args=["-ffast-math"] if platform.system() != "Windows" else [],
    )
]

setuptools.setup(
    name="hybrid_hawkes_exp_cython",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)
