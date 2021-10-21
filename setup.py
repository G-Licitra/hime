import codecs
import os.path

from setuptools import find_packages
from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname("__file__"))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


install_requires = [
    "pandas=1.3.1",
    "casadi>=3.5.5",
    "seaborn>=0.11.0",
    "scikit-learn>=0.22.1",
]

tests_require = [
    "pytest>=6.1.1",
    "pytest-cov>=2.1.0",
    "pytest-mock>=3.5.0",
    "pytest-mpl>=0.12",
]

setup_requires: list = []

packages = find_packages()

setup(
    name="t.b.d.",
    version=get_version("hime/__init__.py"),
    description="lmm",
    url="https://github.com/G-Licitra/robust-mixed-effect-model",
    author="GLicitra, jameshtwose",
    author_email="gianni.licitra7@gmail.com",
    license="t.b.d.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
)