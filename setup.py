#!/usr/bin/env python
"""
Setup configuration for Informer-Based LIBS analysis package.

This package accompanies the paper:
    Walidain, B., Idris, N., Saddami, K., Yuzza, N., & Mitaphonna, R. (2025).
    "Informer-Based LIBS for Qualitative Multi-Element Analysis of an Aceh Traditional Herbal Medicine."
    IOP Conference Series: Earth and Environmental Science, AIC 2025. (in press)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
def read_requirements(filename):
    filepath = Path(__file__).parent / filename
    if not filepath.exists():
        return []
    return [
        line.strip()
        for line in filepath.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="informer-libs-aceh",
    version="1.0.0",
    author="Birrul Walidain, Nasrullah Idris, Khairun Saddami, Natasya Yuzza, Rara Mitaphonna",
    author_email="nasrullah.idris@usk.ac.id",
    maintainer="Birrul Walidain",
    description="Informer-based deep learning for qualitative multi-element LIBS analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/birrulwaldain/informer-libs-aceh",
    project_urls={
        "Bug Tracker": "https://github.com/birrulwaldain/informer-libs-aceh/issues",
        "Documentation": "https://github.com/birrulwaldain/informer-libs-aceh",
        "Source Code": "https://github.com/birrulwaldain/informer-libs-aceh",
        "Paper": "https://iopscience.iop.org/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "informer-libs=app.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "app": [
            "ui/*.py",
            "core/*.py",
        ],
    },
    keywords=[
        "LIBS",
        "Laser-Induced Breakdown Spectroscopy",
        "Informer",
        "deep learning",
        "multi-element analysis",
        "Aceh traditional medicine",
        "Saha-Boltzmann",
        "element detection",
        "scientific software",
    ],
    zip_safe=False,
)

