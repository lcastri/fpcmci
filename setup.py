from setuptools import setup

INSTALL_REQUIRES = ["numpy==1.21.5", "scipy==1.8.0", "tigramite==5.1.0.3"]
EXTRA_REQUIRES = {"all": ["matplotlib==3.6.1",
                          "netgraph==4.10.1",
                          "networkx==2.8.6",
                          "pandas==1.5.0",
                          "ruptures==1.1.7",
                          "scikit_learn==1.1.3",
                          "torch>=1.11.0",       # GPDC torch version
                          "gpytorch>=1.4",       # GPDC gpytorch version
                          "dcor>=0.5.3",         # GPDC distance correlation version              
                         ]
                }

setup(
    name = 'fpcmci',
    version = '1.0.0.0',    
    description = 'A example Python package',
    url = 'https://github.com/lcastri/fpcmci',
    author = 'Luca Castri',
    author_email = 'lucacastri94@gmail.com',
    packages = ['fpcmci', "fpcmci.preprocessing", "fpcmci.utilities", "fpcmci.selection_methods"],
    install_requires = INSTALL_REQUIRES,
    extras_require = EXTRA_REQUIRES,

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',  
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)