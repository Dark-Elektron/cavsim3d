
import os
import sys
from setuptools import setup, find_packages

# -*- coding: utf-8 -*-


with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
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
    python_requires='>=3.9,<3.12',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'tqdm',
        'termcolor',
        'h5py',
        'gmsh',
        'ngsolve',
        'ipython',
        'ipywidgets',
    ],
    extras_require={
        "full": [
            "ngsolve",
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
