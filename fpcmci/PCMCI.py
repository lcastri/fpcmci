import copy
import pickle
from tigramite.pcmci import PCMCI as VAL
from tigramite.independence_tests import CondIndTest
import tigramite.data_processing as pp
import numpy as np
from fpcmci.CPrinter import CPLevel, CP
from fpcmci.preprocessing.data import Data
import fpcmci.basics.utils as utils
from fpcmci.causal_graph import *
from fpcmci.basics.constants import *


class PCMCI():
    """
    PCMCI class.

    PCMCI works with FSelector in order to find the causal 
    model starting from a prefixed set of variables and links.
    """
    def __init__(self, data: Data, alpha, min_lag, max_lag, val_condtest: CondIndTest, resfolder, verbosity: CPLevel):
        """
        PCMCI class constructor

        Args:
            data (Data): data to analyse
            alpha (float): significance level
            min_lag (int): minimum time lag
            max_lag (int): maximum time lag
            val_condtest (CondIndTest): validation method
            resfolder (str): result folder. If None then the results are not saved.
            verbosity (CPLevel): verbosity level
        """
        self.data = data
        self.alpha = alpha
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.result = None
        self.dependencies = None
        self.val_method = None
        self.val_condtest = val_condtest
        self.verbosity = verbosity.value

        self.respath = None
        self.dag_path = None
        self.ts_dag_path = None
        if resfolder is not None:
            self.respath, self.dag_path, self.ts_dag_path = utils.get_validatorpaths(resfolder)  
        

    def run(self, selected_links = None):
        """
        Run causal discovery algorithm

        Returns:
            (dict): estimated causal model
        """
        CP.info('\n')
        CP.info(DASH)
        CP.info("Running Causal Discovery Algorithm")

        # build tigramite dataset
        vector = np.vectorize(float)
        data = vector(self.data.d)
        dataframe = pp.DataFrame(data = data,
                                 var_names = self.data.pretty_features)
        
        # init and run pcmci
        self.val_method = VAL(dataframe = dataframe,
                                cond_ind_test = self.val_condtest,
                                verbosity = self.verbosity)

        self.result = self.val_method.run_pcmci(selected_links = selected_links,
                                                tau_max = self.max_lag,
                                                tau_min = self.min_lag,
                                                pc_alpha = 0.05)
        
        self.result['var_names'] = self.data.pretty_features
        # apply significance level
        self.result['graph'] = self.__apply_alpha()
        self.dependencies = self.__PCMCIres_converter()
        return self.__return_parents_dict()
    
    
    def build_ts_dag(self,
                     min_width,
                     max_width,
                     min_score,
                     max_score,
                     node_size,
                     node_color,
                     edge_color,
                     font_size):
        """
        Saves timeseries dag plot if resfolder is set otherwise it shows the figure
        
        Args:
            min_width (int): minimum linewidt
            max_width (int): maximum linewidth
            min_score (int): minimum score range
            max_score (int): maximum score range
            node_size (int): node size
            node_color (str): node color
            edge_color (str): edge color 
            font_size (int): font size
        """
        
        # # convert to dictionary
        # res = self.__PCMCIres_converter()
        
        # # filter only dependencies
        # tmp_res = {k: res[k] for k in self.data.pretty_features}
        # res = tmp_res
        
        ts_dag(self.dependencies, 
               tau = self.max_lag,
               min_width = min_width,
               max_width = max_width,
               min_score = min_score,
               max_score = max_score,
               node_size = node_size,
               node_color = node_color,
               edge_color = edge_color,
               font_size = font_size,
               save_name = self.ts_dag_path)


    def build_dag(self,
                  node_layout,
                  min_width,
                  max_width,
                  min_score,
                  max_score,
                  node_size,
                  node_color,
                  edge_color,
                  font_size,
                  label_type):
        """
        Saves dag plot if resfolder is set otherwise it shows the figure
        
        Args:
            node_layout (str): node_layout
            min_width (int): minimum linewidth
            max_width (int): maximum linewidth
            min_score (int): minimum score range
            max_score (int): maximum score range
            node_size (int): node size
            node_color (str): node color
            edge_color (str): edge color
            font_size (int): font size
            label_type (LabelType, optional): enum to set whether to show the lag time (LabelType.Lag) or the strength (LabelType.Score) of the dependencies on each link/node or not showing the labels (LabelType.NoLabels). Default LabelType.Lag.

        """               
        
        # # convert to dictionary
        # res = self.__PCMCIres_converter()
        
        # # filter only dependencies
        # tmp_res = {k: res[k] for k in self.data.pretty_features}
        # res = tmp_res
        
        dag(self.dependencies,
            node_layout = node_layout,
            min_width = min_width,
            max_width = max_width,
            min_score = min_score,
            max_score = max_score,
            node_size = node_size,
            font_size = font_size,
            node_color = node_color,
            edge_color = edge_color,
            label_type = label_type,
            save_name = self.dag_path)
        
        
    def save_result(self):
        """
        Save causal discovery results as pickle file if resfolder is set
        """
        if self.respath is not None:
            res = dict()
            res['dependencies'] = copy.deepcopy(self.dependencies)
            # res = copy.deepcopy(self.result)
            res['alpha'] = self.alpha
            res['var_names'] = self.data.pretty_features
            res['dag_path'] = self.dag_path
            res['ts_dag_path'] = self.ts_dag_path
            with open(self.respath, 'wb') as resfile:
                pickle.dump(res, resfile)
        
        
    def __return_parents_dict(self):
        """
        Returns dictionary of parents sorted by val_matrix filtered by alpha

        Returns:
            (dict): Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing estimated parents.
        """
        graph = self.result['graph']
        val_matrix = self.result['val_matrix']
        p_matrix = self.result['p_matrix']
        
        # Initialize the return value
        parents_dict = dict()
        for j in range(self.data.N):
            # Get the good links
            good_links = np.argwhere(graph[:, j, 1:] == "-->")
            # Build a dictionary from these links to their values
            links = {(i, -tau - 1): np.abs(val_matrix[i, j, abs(tau) + 1]) 
                     for i, tau in good_links if p_matrix[i, j, abs(tau) + 1] <= self.alpha}
            # Sort by value
            parents_dict[j] = sorted(links, key=links.get, reverse=True)
        
        return parents_dict
    
    
    def __PCMCIres_converter(self):
        """
        Re-elaborates the PCMCI result in a new dictionary

        Returns:
            (dict): pcmci result re-elaborated
        """
        res_dict = {f:list() for f in self.result['var_names']}
        N, lags = self.result['graph'][0].shape
        for s in range(len(self.result['graph'])):
            for t in range(N):
                for lag in range(lags):
                    if self.result['graph'][s][t,lag] == '-->':
                        res_dict[self.result['var_names'][t]].append({SOURCE : self.result['var_names'][s],
                                                                      SCORE : self.result['val_matrix'][s][t,lag],
                                                                      PVAL : self.result['p_matrix'][s][t,lag],
                                                                      LAG : lag})
        return res_dict
    

    def __apply_alpha(self):
        """
        Applies alpha threshold to the pcmci result

        Returns:
            (ndarray): graph filtered by alpha 
        """
        mask = np.ones(self.result['p_matrix'].shape, dtype='bool')
        
        # Set all p-values of absent links to 1.
        self.result['p_matrix'][mask==False] == 1.
        
        # Threshold p_matrix to get graph
        graph_bool = self.result['p_matrix'] <= self.alpha
        
        # Convert to string graph representation
        graph = self.__convert_to_string_graph(graph_bool)
        
        return graph
    
    
    def __convert_to_string_graph(self, graph_bool):
        """
        Converts the 0,1-based graph returned by PCMCI to a string array
        with links '-->'

        Args:
            graph_bool (array): 0,1-based graph array output by PCMCI

        Returns:
            (array): graph as string array with links '-->'.
        """

        graph = np.zeros(graph_bool.shape, dtype='<U3')
        graph[:] = ""
        # Lagged links
        graph[:,:,1:][graph_bool[:,:,1:]==1] = "-->"
        # Unoriented contemporaneous links
        graph[:,:,0][np.logical_and(graph_bool[:,:,0]==1, 
                                    graph_bool[:,:,0].T==1)] = "o-o"
        # Conflicting contemporaneous links
        graph[:,:,0][np.logical_and(graph_bool[:,:,0]==2, 
                                    graph_bool[:,:,0].T==2)] = "x-x"
        # Directed contemporaneous links
        for (i,j) in zip(*np.where(
            np.logical_and(graph_bool[:,:,0]==1, graph_bool[:,:,0].T==0))):
            graph[i,j,0] = "-->"
            graph[j,i,0] = "<--"
        return graph