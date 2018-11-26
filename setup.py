from setuptools import setup

setup(name='mocha',
      version='0.1',
      description='A toolbox for deeplearning',
      url='https://github.com/laurentgrenier/mocha',
      author='Laurent Grenier',
      author_email='laurent@mooke.io',
      license='MIT',
      packages=['mocha'],
      install_requires=[
            'pandas',
            'numpy',
            'matplotlib',
            'scikit-learn',
            'seaborn',
            'tensorflow-gpu',
            'pymongo'
      ],
      zip_safe=False)

