from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='pypce',
      version='1.0',
      description='pce class for regression',
      long_description='TBD',
      url='TBD',
      author='K. Chowdhary',
      author_email='kchowdh@sandia.gov',
      license='MIT',
      packages=['pypce'],
      test_suite='nose.collector',
      tests_required=['nose'],
      install_requires=[
          'numpy',
          'sklearn',
          'tqdm',
      ],
      include_package_data=True,
      zip_safe=False)