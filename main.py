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


if __name__ == '__main__':   
    alpha = 0.05
    min_lag = 1
    max_lag = 1
    
    np.random.seed(1)
    nsample = 500
    nfeature = 6
    d = np.random.random(size = (nsample, nfeature))
    for t in range(max_lag, nsample):
        d[t, 0] += 5 * d[t-1, 1] + 7 * d[t-1, 2]
        d[t, 2] += 1.5 * d[t-1, 1]
        d[t, 3] += d[t-1, 3] + 1.35 + np.random.randn()
        d[t, 4] += d[t-1, 4] + 1.57 * d[t-1, 5]
        
    df = Data(d)
    FS = FSelector(df, 
                   alpha = alpha, 
                   min_lag = min_lag, 
                   max_lag = max_lag, 
                   sel_method = TE(TEestimator.Gaussian), 
                   val_condtest = GPDC(significance = 'analytic', gp_params = None),
                   verbosity = CPLevel.DEBUG,
                   neglect_only_autodep = False,
                   resfolder = 'example')
    
    selector_res = FS.run()
    FS.dag(show_edge_labels = True, label_type = LabelType.Score)

    
    

