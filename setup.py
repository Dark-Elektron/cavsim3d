# -*- coding: utf-8 -*-
import os
import sys

from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
    readme = f.read()

with open(os.path.join(os.path.dirname(__file__), 'LICENSE')) as f:
    license = f.read()

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
project_root = os.path.abspath(os.path.join(current_dir, 'cavsim3d'))

# Insert the project root into the Python search path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

setup(
    name='cavsim3d',
    version='29.11.2025',
    description='A set of python codes for 3D rf structure analysis.',
    long_description=readme,
    author='Sosoho-Abasi Udongwo',
    author_email='numurho@gmail.com',
    url=r'https://github.com/Dark-Elektron/cavsim3d',
    license=license,
    python_requires='>=3.0, <4',
    install_requires=[
        'matplotlib>=3.8.4',
        'pandas>=2.2.2',
        'numpy==1.26.4',
        'scipy>=1.13.1',
        'psutil>=6.0.0',
        'paretoset',
        'ipython',
        'tqdm>=4.66.4',
        'setuptools>=72.1.0',
        'termcolor>=2.1.0',
        'ngsolve==6.2.2402',
        'openpyxl'
    ],
    extras_require={
        "ngsolve": ["ngsolve"],  # Optional, user must install separately
        "occ": ["pythonocc-core"],
        "sparse": ["scikit-sparse"],
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
    },
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Topic :: Scientific/Engineering :: Physics",
        ],
)
