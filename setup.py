from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Detects automatically NPF events in particle size distribution data',
    author='gfogwill',
    license='MIT',
    entry_points={"console_scripts": ["dmps = src.cli:cli"]},
)
