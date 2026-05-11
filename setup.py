import platform
import setuptools
import numpy

libraries = ["m"] if platform.system() != "Windows" else []
extra_compile_args = ["-ffast-math"] if platform.system() != "Windows" else []

ext_modules = [
    setuptools.Extension('mpoints.hybrid_hawkes_exp_cython',
                         sources=['mpoints/hybrid_hawkes_exp_cython.c'],
                         libraries=libraries,
                         extra_compile_args=extra_compile_args,
                         )
]

setuptools.setup(
    name="mpoints",
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)
