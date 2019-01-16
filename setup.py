from setuptools import setup, find_packages

setup(name='keras_dataloader',

      version='0.1-SNAPSHOT',

      url='https://github.com/GlassyWing/keras_dataloader',

      license='Apache License 2.0',

      author='Manlier',

      author_email='dengjiaxim@gmail.com',

      description='Dataloader for keras',

      packages=find_packages(exclude=['tests', 'examples']),

      package_data={'keras_dataloader': ['*.*', 'checkpoints/*', 'config/*']},

      long_description=open('README.md', encoding="utf-8").read(),

      zip_safe=False,

      python_requires='>=3.6.0',

      install_requires=['keras'],

      )
