from setuptools import setup


requires = [
    'matplotlib>=3.3.2',
    'numpy>=1.20.0',
    'numba>=0.51.2',
    'pandas>=1.1.4',
    'scipy>=1.5.2'
]


setup(
    name='polaris',
    version='0.1.0',
    description='Astrodynamics library in Python',
    url='https://github.com/Yuricst/polaris',
    author='Yuri Shimane',
    author_email='yuri.shimane@gmail.com',
    license='MIT',
    packages=[
        'polaris',
        'polaris.Propagator',
        'polaris.R3BP',
        'polaris.Keplerian',
        'polaris.Coordinates',
    ],
    install_requires=requires,
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
)

