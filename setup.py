from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='effective_quadratures',
      version='0.1.1',
      description='Set of codes for polynomial approximation',
      long_description=readme(),
      classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
      ],
      keywords='polynomial chaos effective quadratures polynomial approximations gradients',
      url='https://github.com/psesh/Effective-Quadratures',
      author='Pranay Seshadri',
      author_email='ps583@cam.ac.uk',
      license='MIT',
      packages=['effective_quadratures'],
      install_requires=[
          'numpy',
          'scipy >= 0.15.0',
          'matplotlib'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
