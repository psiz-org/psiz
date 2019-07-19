"""Setup file."""
from setuptools import setup


def readme():
    """Read in README file."""
    with open('README.rst') as f:
        return f.read()

setup(
    name='psiz',
    version='0.1.0',
    description='Toolbox for inferring psychological embeddings.',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    author='Brett D. Roads',
    author_email='brett.roads@gmail.com',
    license='Apache Licence 2.0',
    packages=['psiz'],
    install_requires=[
        'numpy', 'scipy', 'pandas', 'scikit-learn', 'h5py', 'matplotlib',
        'tensorflow-probability'
    ],
    include_package_data=True,
    url='https://github.com/roads/psiz',
    download_url='https://github.com/roads/psiz/archive/v0.1.0.tar.gz'
)
