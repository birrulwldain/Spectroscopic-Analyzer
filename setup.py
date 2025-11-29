#!/usr/bin/env python
"""
Setup configuration for Spectroscopic Analyzer package.
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
    name="spectroscopic-analyzer",
    version="1.0.0",
    author="Birrulwaldi Nurdin",
    author_email="birrulwaldi@example.com",
    description="AI-powered LIBS spectroscopic analysis and element detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/spectroscopic-analyzer",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/spectroscopic-analyzer/issues",
        "Documentation": "https://github.com/yourusername/spectroscopic-analyzer/docs",
        "Source Code": "https://github.com/yourusername/spectroscopic-analyzer",
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
            "spectroscopic-analyzer=app.main:main",
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
        "spectroscopy",
        "deep learning",
        "element detection",
        "scientific software",
    ],
    zip_safe=False,
)

