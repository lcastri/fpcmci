from tigramite.independence_tests import GPDC
from fpcmci.CPrinter import CPLevel
from fpcmci.FSelector import FSelector
from fpcmci.preprocessing.data import Data
from fpcmci.preprocessing.subsampling_methods.Static import Static
from fpcmci.preprocessing.subsampling_methods.SubsamplingMethod import SubsamplingMethod
from fpcmci.preprocessing.subsampling_methods.WSDynamic import EntropyBasedDynamic
from fpcmci.preprocessing.subsampling_methods.WSFFTStatic import EntropyBasedFFTStatic
from fpcmci.preprocessing.subsampling_methods.WSStatic import EntropyBasedStatic
from fpcmci.selection_methods.TE import TE, TEestimator
import numpy as np
from fpcmci.utilities.constants import LabelType

from time import time
from datetime import timedelta

if __name__ == '__main__':   
    alpha = 0.05
    min_lag = 1
    max_lag = 1
    
    np.random.seed(1)
    nsample = 1500
    nfeature = 6
    d = np.random.random(size = (nsample, nfeature))
    for t in range(max_lag, nsample):
        d[t, 0] += 5 * d[t-1, 1] + 7 * d[t-1, 2]
        d[t, 2] += 1.5 * d[t-1, 1]
        d[t, 3] += d[t-1, 3] + 1.35 + np.random.randn()
        d[t, 4] += d[t-1, 4] + 0.55 * d[t-1, 5]

    df = Data(d)
    start = time()
    FS = FSelector(df, 
                   alpha = alpha, 
                   min_lag = min_lag, 
                   max_lag = max_lag, 
                   sel_method = TE(TEestimator.Gaussian), 
                   val_condtest = GPDC(significance = 'analytic', gp_params = None),
                   verbosity = CPLevel.DEBUG,
                   neglect_only_autodep = False,
                   resfolder = 'ex_PCMCI')
    
    selector_res = FS.run_validator()
    elapsed_PCMCI = time() - start
    print(str(timedelta(seconds = elapsed_PCMCI)))
    FS.dag(show_edge_labels = False, label_type = LabelType.Score)


    
    

