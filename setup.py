"""Setup file."""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='psiz',
    version='0.5.0',
    description='Toolbox for inferring psychological embeddings.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    author='Brett D. Roads',
    author_email='brett.roads@gmail.com',
    license='Apache Licence 2.0',
    packages=['psiz'],
    python_requires='>=3.5, <3.9',
    install_requires=[
        'tensorflow==2.4', 'tensorflow-probability==0.11.0', 'pandas',
        'scikit-learn', 'matplotlib', 'pillow', 'imageio'
    ],
    include_package_data=True,
    url='https://github.com/roads/psiz',
    download_url='https://github.com/roads/psiz/archive/v0.4.2.tar.gz'
)
