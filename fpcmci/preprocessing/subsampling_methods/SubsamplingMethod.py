from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd


class SSMode(Enum):
    WSDynamic = 'Dynamic-size moving window'
    WSStatic = 'Static-size moving window'
    WSFFTStatic = 'FFT static-size moving window'
    Static = 'Static'
    

class SubsamplingMethod(ABC):
    def __init__(self, ssmode: SSMode):
        self.ssmode = ssmode
        self.df = None

    
    def initialise(self, dataframe: pd.DataFrame):
        self.df = dataframe


    @abstractmethod
    def run(self):
        """
        Run subsampler
        """
        pass
