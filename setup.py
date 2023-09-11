# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from pathlib import Path

from setuptools import setup

__version__ = "0.0.1"


def read_requirements(file):
    pwd = Path(__file__).parent.resolve()
    txt = (pwd / file).read_text(encoding="utf-8").split("\n")

    def remove_comments(line: str):
        return len(line) > 0 and not line.startswith(("#", "-"))

    return list(filter(remove_comments, txt))


install_requires = read_requirements("requirements_core.txt")
cpu_requires = read_requirements("requirements_cpu.txt")
ipu_requires = read_requirements("requirements_ipu.txt")
test_requires = read_requirements("requirements_test.txt")

# url = "https://github.com/graphcore-research/jax-experimental/releases/latest/download/"
# ipu_requires = [
#     f"jaxlib @ {url}jaxlib-0.3.15-cp38-none-manylinux2014_x86_64.whl",
#     f"jax @ {url}jax-0.3.16-py3-none-any.whl",
# ]

setup(
    name="pyscf-ipu",
    version=__version__,
    description="PySCF on IPU",
    long_description="file: README.md",
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    author="Graphcore Research",
    author_email="contact@graphcore.ai",
    url="https://github.com/graphcore-research/pyscf-ipu",
    project_urls={
        "Code": "https://github.com/graphcore-research/pyscf-ipu",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=install_requires,
    extras_require={"cpu": cpu_requires, "ipu": ipu_requires, "test": test_requires},
    python_requires=">=3.8",
    packages=["pyscf_ipu"],
    entry_points={"console_scripts": ["nanoDFT=pyscf_ipu.nanoDFT:main"]},
)
