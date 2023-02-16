import codecs
import os
import pathlib

from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()


def read(rel_path):
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="fastrag",
    author="Intel Labs",
    version=get_version("fastrag/__init__.py"),
    description="A research framework for building and evaluating neural information retrieval and generative models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8, <4",
)
