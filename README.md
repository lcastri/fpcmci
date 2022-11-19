# FPCMCI - Filtered PCMCI

Extension of the state-of-the-art causal discovery method PCMCI augmented with a feature-selection method based on Transfer Entropy, that is able to identify the correct subset of variables involved in the causal analysis, starting from a prefixed set of them.


## Why FPCMCI?

Current state-of-the-art causal discovery approaches suffer in terms of speed and accuracy of the causal analysis when the process to be analysed is composed by a large number of features. FPCMCI is able to select the most meaningful features from a set of variables and build a causal model from such selection. To this end, the causal analysis results **faster** and **more accurate**.

In the following it is presented an example showing a comparison between causal models obtained by [PCMCI](https://github.com/jakobrunge/tigramite) and FPCMCI causal discovery algorithms on the same data. The latter have been created as follows:

``` python
min_lag = 1
max_lag = 1
np.random.seed(1)
nsample = 1500
nfeature = 6

d = np.random.random(size = (nsample, feature))
for t in range(max_lag, nsample):
  d[t, 0] += 2 * d[t-1, 1] + 3 * d[t-1, 3]
  d[t, 2] += 1.1 * d[t-1, 1]**2
  d[t, 3] += d[t-1, 3] * d[t-1, 2]
  d[t, 4] += d[t-1, 4] + d[t-1, 5] * d[t-1, 0]
```

Causal Model by PCMCI       |  Causal Model by FPCMCI 
:-------------------------:|:-------------------------:
![](https://github.com/lcastri/fpcmci/raw/main/images/PCMCI_example.png "Causal model by PCMCI")  |  ![](https://github.com/lcastri/fpcmci/raw/main/images/FPCMCI_example.png "Causal model by FPCMCI")
Execution time ~ 6min 50sec | Execution time ~ 2min 45sec

The causal analysis performed by the **FPCMCI** results not only faster but also more accurate. Indeed, the causal model derived by the FPCMCI agrees with the structure of the system of equations, instead the onoe derived by the PCMCI presents spurious links:
* $X_2$ &rarr; $X_4$
* $X_2$ &rarr; $X_5$


## Citation

If you found this useful for your work, please cite these papers:
```
@article{ghidoni2022human,
  title={From Human Perception and Action Recognition to Causal Understanding of Human-Robot Interaction in Industrial Environments},
  author={Ghidoni, Stefano and Terreran, Matteo and Evangelista, Daniele and Menegatti, Emanuele and Eitzinger, Christian and Villagrossi, Enrico and Pedrocchi, Nicola and Castaman, Nicola and Malecha, Marcin and Mghames, Sariah and others},
  year={2022}
}
```
```
@inproceedings{castri2022causal,
    title={Causal Discovery of Dynamic Models for Predicting Human Spatial Interactions},
    author={Castri, Luca and Mghames, Sariah and Hanheide, Marc and Bellotto, Nicola},
    booktitle={International Conference on Social Robotics (ICSR)},
    year={2022},
}
```


## Requirements

* matplotlib==3.6.1
* netgraph==4.10.2
* networkx==2.8.6
* numpy==1.21.5
* pandas==1.5.0
* ruptures==1.1.7
* scikit_learn==1.1.3
* scipy==1.8.0
* setuptools==56.0.0
* tigramite==5.1.0.3


## Installation
Before installing the FPCMCI package, you need to install the [IDTxl package](https://github.com/pwollstadt/IDTxl) used for the feature-selection process, following the guide described [here](https://github.com/pwollstadt/IDTxl/wiki/Installation-and-Requirements). Once complete, you can install the current release of `FPCMCI` with:
``` shell
pip install fpcmci
```

## Documentation
Coming soon..