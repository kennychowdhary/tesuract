from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tesuract",
    version="0.2.0",
    description="Tensor surrogate construction methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kennychowdhary/tesuract",
    author="Kenny Chowdhary",
    author_email="kchowdh@sandia.gov",
    license="BSD3",
    packages=["tesuract"],
    test_suite="nose.collector",
    tests_required=["nose"],
    install_requires=["numpy", "sklearn", "scipy", "tqdm"],
    include_package_data=True,
    zip_safe=False,
)
