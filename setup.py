from setuptools import setup
from os import path

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Financial and Insurance Industry",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent"
]

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="stockml",
    version="0.0.1",
    description=" Common financial technical indicators implemented in Pandas.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["technical analysis", "ta", "pandas", "finance", "numpy", "analysis"],
    url="https://github.com/davidqchuang/stockanalysis-ml",
    project_urls={
      "Homepage": "https://github.com/davidqchuang/stockanalysis-ml",
      "Issues": "https://github.com/davidqchuang/stockanalysis-ml/issues"
    },
    author="dqchuang",
    author_email="dqchuang@outlook.com",
    license="MIT",
    package_dir = {"": "src"},
    install_requires=["pandas", "numpy"],
    license_files=["LICENSE"]
)