"""Setup file."""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='psiz',
    version='0.5.1',
    description='Toolbox for inferring psychological embeddings.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/roads/psiz',
    author='Brett D. Roads',
    author_email='brett.roads@gmail.com',
    license='Apache Licence 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='psychology, cognitive science',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.6, <3.9',
    install_requires=[
        'tensorflow==2.4', 'tensorflow-probability==0.11.0', 'pandas',
        'scikit-learn', 'matplotlib', 'pillow', 'imageio'
    ],
    project_urls={
        'Documentation': 'https://psiz.readthedocs.io/en/latest/',
        'Source': 'https://github.com/roads/psiz',
        'Tracker': 'https://github.com/roads/psiz/issues',
    },
    include_package_data=True,
)
