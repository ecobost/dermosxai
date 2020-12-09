#!/usr/bin/env python3
from setuptools import setup

setup(
    name='dermosxai',
    version='0.0.1',
    description='Explainability methods applied to dermatoscopic images',
    author='Erick Cobos',
    author_email='ecobos@tuebingen.mpg.de',
    license='MIT',
    url='https://github.com/ecobost/dermosxai',
    #keywords= '2p 3d GCaMPs soma segmentation stack',
    packages=['dermosxai'],
    install_requires=['torch', 'numpy', 'pandas', 'scikit-learn', 'h5py'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
