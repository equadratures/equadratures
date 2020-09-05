from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='equadratures',
      version='9.0.1',
      description='Polynomial approximations',
      long_description=readme(),
      classifiers=[
	'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
      ],
      keywords='polynomial chaos effective quadratures polynomial approximations gradients',
      url='https://github.com/Effective-Quadratures/Effective-Quadratures',
      author='Pranay Seshadri, Nicholas Wong, Irene Virdis, James Gross, Joe Zhou, Ashley Scillitoe, Barney Hill, Bryn Ubald',
      license='LPGL-2.1',
      packages=['equadratures', 'equadratures.distributions', 'equadratures.sampling_methods'],
      install_requires=[
          'numpy',
          'scipy >= 0.15.0',
          'matplotlib'
      ],
      extras_require={
          "cvxpy":  ['cvxpy>=1.1'],
          },
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
