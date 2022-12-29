import copy
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tigramite.independence_tests import CondIndTest
import sys
from fpcmci.selection_methods.SelectionMethod import SelectionMethod
from fpcmci.CPrinter import CPLevel, CP
from fpcmci.basics.constants import *
from fpcmci.basics.logger import Logger
import fpcmci.basics.utils as utils
from fpcmci.PCMCI import PCMCI
from fpcmci.preprocessing.data import Data 


class FPCMCI():
    """
    FPCMCI class.

    FPCMCI is a causal feature selector framework for large-scale time series
    datasets. Sarting from a Data object and it selects the main features
    responsible for the evolution of the analysed system. Based on the selected features,
    the framework outputs a causal model.
    """

    def __init__(self, 
                 data: Data, 
                 min_lag, max_lag, 
                 sel_method: SelectionMethod, val_condtest: CondIndTest, 
                 verbosity: CPLevel, 
                 alpha = 0.05, 
                 resfolder = None,
                 neglect_only_autodep = False):
        """
        FPCMCI class contructor

        Args:
            data (Data): data to analyse
            min_lag (int): minimum time lag
            max_lag (int): maximum time lag
            sel_method (SelectionMethod): selection method
            val_condtest (CondIndTest): validation method
            verbosity (CPLevel): verbosity level
            alpha (float, optional): significance level. Defaults to 0.05.
            resfolder (string, optional): result folder to create. Defaults to None.
            neglect_only_autodep (bool, optional): Bit for neglecting variables with only autodependency. Defaults to False.
        """
        
        self.data = data
        self.alpha = alpha
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.sel_method = sel_method
        self.filter_dependencies = None
        self.o_filter_dependencies = None
        self.causal_model = None
        self.result = None
        self.neglect_only_autodep = neglect_only_autodep

        self.dependency_path = None
        if resfolder is not None:
            utils.create_results_folder()
            logpath, self.dependency_path = utils.get_selectorpath(resfolder)
            sys.stdout = Logger(logpath)
        
        self.validator = PCMCI(data, alpha, min_lag, max_lag, val_condtest, resfolder, verbosity)       
        CP.set_verbosity(verbosity)


    def run_filter(self):
        """
        Run filter method
        """
        CP.info("\n")
        CP.info(DASH)
        CP.info("Selecting relevant features among: " + str(self.data.features))
        CP.info("Selection method: " + self.sel_method.name)
        CP.info("Significance level: " + str(self.alpha))
        CP.info("Max lag time: " + str(self.max_lag))
        CP.info("Min lag time: " + str(self.min_lag))
        CP.info("Data length: " + str(self.data.T))

        self.sel_method.initialise(self.data, self.alpha, self.min_lag, self.max_lag)
        self.filter_dependencies = self.sel_method.compute_dependencies()
        self.o_filter_dependecies = copy.deepcopy(self.filter_dependencies)


    def run_pcmci(self):
        """
        Run PCMCI
        
        Returns:
            list(str): list of selected variable names
        """
        CP.info("Significance level: " + str(self.alpha))
        CP.info("Max lag time: " + str(self.max_lag))
        CP.info("Min lag time: " + str(self.min_lag))
        CP.info("Data length: " + str(self.data.T))

        # causal model
        self.validator.data = self.data
        self.validator.run()
        self.causal_model = self.validator.dependencies
        
        self.result = self.data.features
        
        self.save_validator_res()
        
        return self.result, self.causal_model

    
    def run(self):
        """
        Run Selector and Validator
        
        Returns:
            list(str): list of selected variable names
        """
        
        self.run_filter()        
            
        # list of selected features based on dependencies
        tmp_sel_features = self.get_selected_features()
        if not tmp_sel_features:
            return self.result

        # shrink dataframe d and dependencies by the selector result
        self.shrink(tmp_sel_features)
        
        # selected links to check by the validator
        selected_links = self.__get_selected_links()
            
        # causal model on selected links
        self.validator.data = self.data
        pcmci_result = self.validator.run(selected_links)
        
        # application of the validator result to the filter_dependencies field
        self.__apply_validator_result(pcmci_result)
        # final causal model
        self.causal_model = self.validator.dependencies
        
        self.result = self.get_selected_features()
        # shrink dataframe d and dependencies by the validator result
        self.shrink(self.result)
        
        self.save_validator_res()
        
        CP.info("\nFeature selected: " + str(self.result))
        return self.result, self.causal_model
    

    def shrink(self, sel_features):
        """
        Wrapper in order to shrink data.d and dependencies

        Args:
            sel_features (list(str)): list of selected features
        """
        self.data.shrink(sel_features)
        self.__shrink_dependencies()
    
    
    def save_validator_res(self):
        """
        Saves dag plot if resfolder has been set otherwise it shows the figure
        """
        if self.result:
            self.validator.save_result()
        else:
            CP.warning("Result impossible to save: no feature selected")
    
    
    def dag(self,
            node_layout = 'dot',
            min_width = 1,
            max_width = 5,
            min_score = 0,
            max_score = 1,
            node_size = 8,
            node_color = 'orange',
            edge_color = 'grey',
            font_size = 12,
            label_type = LabelType.Lag):
        """
        Saves dag plot if resfolder has been set otherwise it shows the figure
        
        Args:
            node_layout (str, optional): Node layout. Defaults to 'dot'.
            min_width (int, optional): minimum linewidth. Defaults to 1.
            max_width (int, optional): maximum linewidth. Defaults to 5.
            min_score (int, optional): minimum score range. Defaults to 0.
            max_score (int, optional): maximum score range. Defaults to 1.
            node_size (int, optional): node size. Defaults to 8.
            node_color (str, optional): node color. Defaults to 'orange'.
            edge_color (str, optional): edge color. Defaults to 'grey'.
            font_size (int, optional): font size. Defaults to 12.
            label_type (LabelType, optional): enum to set whether to show the lag time (LabelType.Lag) or the strength (LabelType.Score) of the dependencies on each link/node or not showing the labels (LabelType.NoLabels). Default LabelType.Lag.
        """
        
        if self.result:
            self.validator.build_dag(node_layout,
                                     min_width, 
                                     max_width,
                                     min_score,
                                     max_score,
                                     node_size,
                                     node_color,
                                     edge_color,
                                     font_size,
                                     label_type)
        else:
            CP.warning("Dag impossible to create: no feature selected")
    
        
    def timeseries_dag(self,
                       min_width = 1,
                       max_width = 5,
                       min_score = 0,
                       max_score = 1,
                       node_size = 8,
                       font_size = 12,
                       node_color = 'orange',
                       edge_color = 'grey'):
        """
        Saves timeseries dag plot if resfolder has been set otherwise it shows the figure
        
        Args:
            min_width (int, optional): minimum linewidth. Defaults to 1.
            max_width (int, optional): maximum linewidth. Defaults to 5.
            min_score (int, optional): minimum score range. Defaults to 0.
            max_score (int, optional): maximum score range. Defaults to 1.
            node_size (int, optional): node size. Defaults to 8.
            node_color (str, optional): node color. Defaults to 'orange'.
            edge_color (str, optional): edge color. Defaults to 'grey'.
            font_size (int, optional): font size. Defaults to 12.
        """
        
        if self.result:
            self.validator.build_ts_dag(min_width,
                                        max_width,
                                        min_score,
                                        max_score,
                                        node_size,
                                        node_color,
                                        edge_color,
                                        font_size)
        else:
            CP.warning("Timeseries dag impossible to create: no feature selected")


    def get_selected_features(self):
        """
        Defines the list of selected variables for d

        Returns:
            list(str): list of selected variable names
        """
        f_list = list()
        for t in self.filter_dependencies:
            sources_t = self.__get_dependencies_for_target(t)
            if self.neglect_only_autodep and self.__is_only_autodep(sources_t, t):
                sources_t.remove(t)
            if sources_t: sources_t.append(t)
            f_list = list(set(f_list + sources_t))
        res = [f for f in self.data.features if f in f_list]

        return res
    
    
    def show_dependencies(self):
        """
        Saves dependencies graph if resfolder is set otherwise it shows the figure
        """
        # FIXME: LAG not considered
        dependencies_matrix = self.__get_dependencies_matrix()

        fig, ax = plt.subplots()
        im = ax.imshow(dependencies_matrix, cmap=plt.cm.Greens, interpolation='nearest', vmin=0, vmax=1, origin='lower')
        fig.colorbar(im, orientation='vertical', label="score")

        plt.xlabel("Sources")
        plt.ylabel("Targets")
        plt.xticks(ticks = range(0, self.data.orig_N), labels = self.data.orig_pretty_features, fontsize = 8)
        plt.yticks(ticks = range(0, self.data.orig_N), labels = self.data.orig_pretty_features, fontsize = 8)
        plt.title("Dependencies")

        if self.dependency_path is not None:
            plt.savefig(self.dependency_path, dpi = 300)
        else:
            plt.show()


    def print_dependencies(self):
        """
        Print dependencies found by the selector
        """
        for t in self.o_filter_dependecies:
            print()
            print()
            print(DASH)
            print("Target", t)
            print(DASH)
            print('{:<10s}{:>15s}{:>15s}{:>15s}'.format('SOURCE', 'SCORE', 'PVAL', 'LAG'))
            print(DASH)
            for s in self.o_filter_dependecies[t]:
                print('{:<10s}{:>15.3f}{:>15.3f}{:>15d}'.format(s[SOURCE], s[SCORE], s[PVAL], s[LAG]))      


    def load_result(self, res_path):
        with open(res_path, 'rb') as f:
            self.validator.result = pickle.load(f)


    def __shrink_dependencies(self):
        """
        Shrinks dependencies based on the selected features
        """
        difference_set = self.filter_dependencies.keys() - self.data.features
        for d in difference_set: del self.filter_dependencies[d]
        

    def __get_dependencies_for_target(self, t):
        """
        Returns list of sources for a specified target

        Args:
            t (str): target variable name

        Returns:
            list(str): list of sources for target t
        """
        return [s[SOURCE] for s in self.filter_dependencies[t]]
    
    
    def __is_only_autodep(self, sources, t):
        """
        Returns list of sources for a specified target

        Args:
            sources (list(str)): list of sources for the selected target
            t (str): target variable name

        Returns:
            (bool): True if sources list contains only the target. False otherwise
        """
        if len(sources) == 1 and sources[0] == t: return True
        return False


    def __get_dependencies_matrix(self):
        """
        Returns a matrix composed by scores for each target

        Returns:
            (np.array): score matrix
        """
        dep_mat = list()
        for t in self.o_filter_dependecies:
            dep_vet = [0] * self.data.orig_N
            for s in self.o_filter_dependecies[t]:
                dep_vet[self.data.orig_features.index(s[SOURCE])] = s[SCORE]
            dep_mat.append(dep_vet)

        dep_mat = np.array(dep_mat)
        inf_mask = np.isinf(dep_mat)
        neginf_mask = np.isneginf(dep_mat)
        max_dep_mat = np.max(dep_mat[(dep_mat != -np.inf) & (dep_mat != np.inf)])
        min_dep_mat = np.min(dep_mat[(dep_mat != -np.inf) & (dep_mat != np.inf)])

        dep_mat[inf_mask] = max_dep_mat
        dep_mat[neginf_mask] = min_dep_mat
        dep_mat = (dep_mat - min_dep_mat) / (max_dep_mat - min_dep_mat)
        return dep_mat


    def __get_selected_links(self):
        """
        Return selected links found by the selector
        in this form: {0: [(0,-1), (2,-1)]}

        Returns:
            (dict): selected links
        """
        sel_links = {self.data.features.index(f):list() for f in self.data.features}
        for t in self.filter_dependencies:
            
            # add links
            for s in self.filter_dependencies[t]:
                sel_links[self.data.features.index(t)].append((self.data.features.index(s[SOURCE]), -s[LAG]))

        return sel_links
    
    
    def __apply_validator_result(self, causal_model):
        """
        Exclude dependencies based on validator result
        It does not overwrite the filter_dependencies' inference/p-values matrix with the ones found by the validator
        """
        list_diffs = list()
        tmp_dependencies = copy.deepcopy(self.filter_dependencies)
        for t in tmp_dependencies:
            for s in tmp_dependencies[t]:
                if (self.data.features.index(s[SOURCE]), -s[LAG]) not in causal_model[self.data.features.index(t)]:
                    list_diffs.append((s[SOURCE], str(s[LAG]), t))
                    self.filter_dependencies[t].remove(s)
        if list_diffs:
            CP.debug(DASH)
            CP.debug("Difference(s)")
            CP.debug(DASH)
            for diff in list_diffs:
                CP.debug("Removing (" + diff[0] + " -" + diff[1] +") --> (" + diff[2] + ")")
    
