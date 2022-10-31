from abc import ABC, abstractmethod
from enum import Enum
from contextlib import contextmanager
import sys, os
from fpcmci.preprocessing.data import Data
from fpcmci.utilities.constants import *


class CTest(Enum):
    Corr = "Correlation"
    MI = "Mutual Information"
    TE = "Transfer Entropy"


@contextmanager
def _suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


class SelectionMethod(ABC):
    def __init__(self, ctest):
        self.ctest = ctest
        self.data = None
        self.alpha = None
        self.min_lag = None
        self.max_lag = None
        self.result = dict()


    @property
    def name(self):
        """
        Returns Selection Method name

        Returns:
            str: Selection Method name
        """
        return self.ctest.value


    def initialise(self, data: Data, alpha, min_lag, max_lag):
        """
        Initialises the selection method

        Args:
            data (Data): Data
            alpha (float): significance threshold
            min_lag (int): min lag time
            max_lag (int): max lag time
        """
        self.data = data
        self.alpha = alpha
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.result = {f:list() for f in self.data.features}


    @abstractmethod
    def compute_dependencies(self) -> dict:
        pass


    def _prepare_ts(self, target, lag, apply_lag = True, consider_autodep = True):
        """
        prepare the dataframe to the analysis

        Args:
            target (str): name target var
            lag (int): lag time to apply
            apply_lag (bool, optional): True if you want to apply the lag, False otherwise. Defaults to True.

        Returns:
            tuple(DataFrame, DataFrame): source and target dataframe
        """
        if not consider_autodep:
            if apply_lag:
                Y = self.data.d[target][lag:]
                X = self.data.d.loc[:, self.data.d.columns != target][:-lag]
            else:
                Y = self.data.d[target]
                X = self.data.d.loc[:, self.data.d.columns != target]
        else:
            if apply_lag:
                Y = self.data.d[target][lag:]
                X = self.data.d[:-lag]
            else:
                Y = self.data.d[target]
                X = self.data.d
        return X, Y


    def _get_sources(self, t):
        """
        Return target sources

        Args:
            t (str): target variable name

        Returns:
            list(str): list of target sources
        """
        return [s[SOURCE] for s in self.result[t]]


    def _add_dependecies(self, t, s, score, pval, lag):
        """
        Adds found dependency from source (s) to target (t) specifying the 
        score, pval and the lag

        Args:
            t (str): target feature name
            s (str): source feature name
            score (float): selection method score
            pval (float): pval associated to the dependency
            lag (int): lag time of the dependency
        """
        self.result[t].append({SOURCE:s, 
                               SCORE:score,
                               PVAL:pval, 
                               LAG:lag})
        str_s = "(" + s + " -" + str(lag) + ")"
        str_arrow = " --> "
        str_t = "(" + t + ")"
        str_score = "|score: " + "{:.3f}".format(score)
        str_pval = "|pval: " + "{:.3f}".format(pval)
        print('{:<20s}{:<10s}{:<10s}{:<20s}{:<20s}'.format(str_s, str_arrow, str_t, str_score, str_pval))
