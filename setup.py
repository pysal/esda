from setuptools import setup, find_packages

from distutils.command.build_py import build_py

with open('README.rst', 'r', encoding='utf8') as file:
    long_description = file.read()

# Get __version__ from giddy/__init__.py without importing the package
# __version__ has to be defined in the first line
with open('esda/__init__.py', 'r') as f:
    exec(f.readline())


setup(name='esda',  # name of package
      version=__version__,
      description='Package with statistics for exploratory spatial data analysis',
      long_description=long_description,
      url='https://github.com/pysal/esda',
      maintainer='Sergio Rey, Wei Kang',
      maintainer_email='sjsrey@gmail.com, weikang9009@gmail.com',
      test_suite='nose.collector',
      py_modules=['esda'],
      python_requires='>3.4',
      tests_require=['nose'],
      keywords='spatial statistics',
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
        ],
      license='3-Clause BSD',
      packages=find_packages(),
      install_requires=['libpysal'],
      zip_safe=False,
      cmdclass={'build.py': build_py})
