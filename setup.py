"""Setup file."""
from setuptools import setup


setup(
    name='psiz',
    version='0.2.1',
    description='Toolbox for inferring psychological embeddings.',
    long_description='See https://github.com/roads/psiz for more information.',
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
    download_url='https://github.com/roads/psiz/archive/v0.2.1.tar.gz'
)
