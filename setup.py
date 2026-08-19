
import os
import sys
from setuptools import setup, find_packages

# -*- coding: utf-8 -*-


with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding="utf-8") as f: #<- important to explictly state format
    readme = f.read()

with open(os.path.join(os.path.dirname(__file__), 'LICENSE')) as f:
    license = f.read()


setup(
    name='cavsim3d',
    version='0.1.0',
    description='A set of python codes for 3D rf structure analysis.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Sosoho-Abasi Udongwo',
    author_email='numurho@gmail.com',
    url=r'https://github.com/Dark-Elektron/cavsim3d',
    license='LGPL',
    python_requires='>=3.9,<3.14',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'tqdm',
        'termcolor',
        'h5py',
        'gmsh',
        'ngsolve==6.2.2506',
        # ngsolve 6.2.2506 links libmkl_rt.so.2 (Linux) / mkl_rt.2.dll (Windows).
        # mkl >= 2026 renamed these to .so.3 / mkl_rt.3.dll, so ngsolve's ctypes
        # preload no longer satisfies its DT_NEEDED -> ImportError on import.
        # macOS ngsolve wheels do not link MKL at all.
        'mkl<2026; platform_system != "Darwin"',
        'ipython',
        'ipywidgets',
    ],
    extras_require={
        "full": [
            "ipython",
            "ipywidgets",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
    packages=find_packages(exclude=('tests', 'docs', 'examples', 
                                      'notebooks', 'scripts', 'site')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
