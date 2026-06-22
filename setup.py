from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='equadratures',
      version='10.1.0',
      description='Polynomial approximations',
      long_description=readme(),
      classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
      ],
      keywords='polynomial chaos effective quadratures polynomial approximations gradients',
      url='https://github.com/Effective-Quadratures/equadratures',
      author='Developers',
      license='LPGL-2.1',
      packages=['equadratures', 'equadratures.distributions', 'equadratures.sampling_methods'],
      install_requires=[
          'numpy',
          'scipy >= 0.15.0',
          'matplotlib',
          'seaborn',
          'requests >= 2.11.1',
          'graphviz'
      ],
      extras_require={
          "cvxpy":  ['cvxpy>=1.1'],
          "networkx":  ['networkx==2.6.3'],
          "torch" : ['torch>=1.7.0'],
          "tensorflow": ['tensorflow==1.15.2'],
          "pymanopt": ['pymanopt'],
          ":python_version == '3.6'": ["tomli<2.3"],
          ":python_version == '3.6'": ["dataclasses"],
          ":python_version <= '3.8'": ["typing_extensions<4.14"]
          },
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
