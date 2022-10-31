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
from time import time
from datetime import timedelta

if __name__ == '__main__':   
    alpha = 0.05
    min_lag = 1
    max_lag = 1
    
    df = Data('data/Exp_1_run_1/agent_10_cut.csv', subsampling = Static(10))
    df = Data(df.d, subsampling = EntropyBasedDynamic(250, 0.05))
    start = time()
    FS = FSelector(df, 
                   alpha = alpha, 
                   min_lag = min_lag, 
                   max_lag = max_lag, 
                   sel_method = TE(TEestimator.Gaussian), 
                   val_condtest = GPDC(significance = 'analytic', gp_params = None),
                   verbosity = CPLevel.DEBUG,
                   neglect_only_autodep = False,
                   resfolder = 'THOR10_onlyPCMCI_withtime')
    
    selector_res = FS.run_validator()
    FS.dag(show_edge_labels = False)
    elapsed = time() - start
    print(str(timedelta(seconds = elapsed)))
    # FS.timeseries_dag()

    
    

