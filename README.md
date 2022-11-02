# FPCMCI - Filtered PCMCI

Extension of the state-of-the-art causal discovery method PCMCI augmented with a feature-selection method based on Transfer Entropy, that is able to identify the correct subset of variables involved in the causal analysis, starting from a prefixed set of them.


# Why FPCMCI?

Current state-of-the-art causal discovery approaches suffer in terms of speed and accuracy of the causal analysis when the process to be analysed is composed by a large number of features. FPCMCI is ble to select the most meaningful features from a set of variables and build a causal model from such selection. To this end, the causal analysis results **faster** and **more accurate**.


# Requirements

* matplotlib==3.6.1
* netgraph==4.10.1
* networkx==2.8.6
* numpy==1.21.5
* pandas==1.5.0
* ruptures==1.1.7
* scikit_learn==1.1.3
* scipy==1.8.0
* setuptools==56.0.0
* tigramite==5.1.0.3


# Installation

Before installing the FPCMCI package, you need to install the [IDTxl package](https://github.com/pwollstadt/IDTxl) used for the feature-selection process, following the guide described [here](https://github.com/pwollstadt/IDTxl/wiki/Installation-and-Requirements). Once complete, you can install the package with:
```
pip install fpcmci
```


# Documentation
Coming soon..


# Quickstart
Coming soon..