# from distutils.core import setup, find_packages
import numpy
from setuptools import setup
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    # Metadata
    name="cluster",
    version="0.1.0",
    description="Fast Non-Parametric Item Response Theory Solver",

    # Package info
    packages=["nirt"],
    include_package_data=True,
    zip_safe=True,
    install_requires=[
        "bottleneck",
        "cython",
        "kmodes",
        "matplotlib",  # For plotting clusters only
        "numpy",
        "pytest",
        "scikit-learn",
    ],
    ext_modules=cythonize(["cntree/*.pyx"], annotate=True),
    package_dir={'cluster': ''},
    include_dirs=[numpy.get_include()],
)
