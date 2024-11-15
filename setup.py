#!/usr/bin/env python3
from pathlib import Path
import re
import setuptools


here = Path(__file__).parent.absolute()
required = [
    r for r in (here / 'requirements.in').read_text().splitlines()
]
version = re.findall(
    r'__version__ *= *[\'"]([^\'"]+)',
    (here / 'pps_mw_training' / '__init__.py').read_text(encoding='utf-8')
)[-1]
long_description = """
    Package for NWCSAF/PPS-MW neural network training.
    TODO: add more text
"""

setuptools.setup(
    name='pps-mw-validation',
    version=version,
    description='Package for NWCSAF/PPS-MW neural network training.',
    author='Bengt Rydberg',
    author_email='bengt.rydberg@smhi.se',
    url='http://nwcsaf.org',
    long_description=long_description,
    license='GPL',
    packages=setuptools.find_packages(),
    python_requires='>=3.9, <4',
    install_requires=required,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'train=scripts.train:cli',
        ],
    }
)
