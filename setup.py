import os
import pathlib

from setuptools import find_namespace_packages, find_packages, setup

here = pathlib.Path(__file__).parent.resolve()


def read(rel_path):
    with open(os.path.join(here, rel_path), "r", encoding="utf-8") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


setup(
    name="fastrag",
    author="Intel Labs",
    version=get_version("fastrag/__init__.py"),
    packages=find_namespace_packages(include=["fastrag*"]),
    description="An Efficient Retrieval Augmentation and Generation Framework for Intel Hardware.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/IntelLabs/fastRAG",
    license="Apache-2.0",
    python_requires=">=3.8, <3.12",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
)
