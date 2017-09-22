from setuptools import setup

from distutils.command.build_py import build_py

setup(name='esda',  # name of package
      version='1.0.1dev',
      description='Package with statistics for exploratory spatial data analysis',
      url='https://github.com/pysal/esda',
      maintainer='Sergio Rey',
      maintainer_email='sjsrey@gmail.com',
      test_suite='nose.collector',
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
        'Programming Language :: Python :: 2.5',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4'
        ],
      license='3-Clause BSD',
      packages=['esda'],
      install_requires=['numpy', 'scipy', 'libpysal'
                        ],
      zip_safe=False,
      cmdclass={'build.py': build_py})
