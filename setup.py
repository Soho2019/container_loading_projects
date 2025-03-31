from setuptools import setup, find_packages

setup(
    name="container_loading",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pulp',
        'scikit-learn',
        'scipy'
    ],
)