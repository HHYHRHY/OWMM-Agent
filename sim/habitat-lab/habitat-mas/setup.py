#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

VERSION = "0.1.0"

def read(file_path, *args, **kwargs):
    with open(file_path, *args, **kwargs) as f:
        content = f.read()
    return content

if __name__ == "__main__":
    setup(
        name="habitat-mas",
        install_requires=read("requirements.txt").strip().split("\n"),
        packages=find_packages(),
        version=VERSION,
        include_package_data=True,
        description="Habitat-MAS: Multi-Agent Systems in Habitat virtual environments",
        long_description=read("README.md", encoding="utf8"),
        long_description_content_type="text/markdown",
        author="Junting Chen",
        license="MIT License",
        # url="https://aihabitat.org",
        # project_urls={
        #     "GitHub repo": "https://github.com/facebookresearch/habitat-mas/",
        #     "Bug Tracker": "https://github.com/facebookresearch/habitat-mas/issues",
        # },
        classifiers=[
            "Intended Audience :: Science/Research",
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Operating System :: MacOS",
            "Operating System :: Unix",
        ],
        # no entry points
    )