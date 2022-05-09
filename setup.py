from setuptools import setup


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name="tesuract",
    version="0.2.0",
    description="Tensor surrogate construction methods",
    long_description="UQ and ML tools for surrogate construction",
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
