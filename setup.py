from setuptools import setup


requires = ["numba>=0.51.2"]


setup(
    name='polaris',
    version='0.1',
    description='Awesome library',
    url='https://github.com/Yuricst/polaris',
    author='Yuri',
    author_email='yuri.shimane@gmail.com',
    license='MIT',
    keywords='astrodynamics',
    packages=[
        "polaris",
        "polaris.Propagator",
        "polaris.R3BP",
        "polaris.Keplerian",
        "polaris.Coordinates",
    ],
    install_requires=requires,
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
)
