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
    "seaborn>=0.11.0",
    "statsmodels>=0.12.0",
    "scikit-learn>=0.23.0",
    "boto3>=1.16.0",
    "requests>=2.24.0",
    # "jupyter>=1.0.0", # stopped reading this in as the meta package is 6 years old now so doesn't support new things
]

tests_require = [
    "pytest>=6.2.0",
    "pytest-cov>=2.11.0",
    "pytest-mock>=3.5.0",
    "pytest-mpl>=0.12",
]

setup_requires: list = []

packages = find_packages()

setup(
    name="romeo",
    version=get_version("romeo/__init__.py"),
    description="Robust Mixed Effect Model",
    url="https://github.com/GLicitra/robust-mixed-effect-model",
    author="GLicitra",
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
)
