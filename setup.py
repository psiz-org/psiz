from setuptools import setup

def readme():
    '''
    Read in README file.
    '''
    with open('README.rst') as f:
        return f.read()

setup(name='psychembed',
      version='0.1.0',
      description='Toolbox for inferring psychological embeddings.',
      long_description=readme(),
      classifiers=[
          'Programming Language :: Python :: 3',
      ],
      author='Brett D. Roads',
      author_email='brett.roads@gmail.com',
      license='GNU GPLv3',
      packages=['psychembed'],
      install_requires=['numpy', 'tensorflow', 'pandas', 'scikit-learn'],
      include_package_data=True,
      )
