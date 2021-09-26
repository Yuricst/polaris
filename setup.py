from setuptools import setup
import os

requires = [
    'matplotlib>=3.3.2',
    'numpy>=1.20.0',
    'numba>=0.51.2',
    'pandas>=1.1.4',
    'scipy>=1.5.2'
]


# long_description for readme
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='astro-polaris',
    version='0.1.3',
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
    long_description=long_description,                                   # extract README file
    long_description_content_type='text/markdown',         # long_descriptionの形式を'text/plain', 'text/x-rst', 'text/markdown'のいずれかから指定
    keywords='astrodynamics python CR3BP',                 # PyPIでの検索用キーワードをスペース区切りで指定
    install_requires=requires,
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
)

