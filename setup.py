#!/usr/bin/env python3
"""
Vizor v1.0 - Local-first Cybersecurity Copilot
Setup and installation configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vizor",
    version="1.0.0",
    author="Vizor Development Team",
    author_email="dev@vizor.local",
    description="Local-first, adaptive cybersecurity copilot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vizor/vizor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vizor=cli.main:app",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.yaml", "*.json"],
        "docs": ["*.md"],
    },
)
