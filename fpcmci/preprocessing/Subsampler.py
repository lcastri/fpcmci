from fpcmci.preprocessing.subsampling_methods.SubsamplingMethod import SubsamplingMethod
import pandas as pd
from matplotlib import gridspec
import pylab as pl
import numpy as np

class Subsampler():

    def __init__(self, 
                 df: pd.DataFrame, 
                 ss_method: SubsamplingMethod):
        self.df = df
        self.ss_method = ss_method
        self.ss_method.initialise(df)


    def subsample(self):
        """
        Sabsamples dataframe

        Returns:
            type: Subsampled dataframe value
        """
        self.result = self.ss_method.run()
        return self.df.values[self.result, :]


    def plot_subsampled_data(self, dpi = 100, show = True):
        """
        Plot dataframe sub-sampled data

        Args:
            dpi (int, optional): image dpi. Defaults to 100.
            show (bool, optional): if True it shows the figure and block the process. Defaults to True.
        """
        n_plot = self.df.shape[1]

        # Create grid
        gs = gridspec.GridSpec(n_plot, 1)

        # Time vector
        T = list(range(0, self.df.shape[0]))

        pl.figure(dpi = dpi)
        for i in range(0, n_plot):
            ax = pl.subplot(gs[i, 0])
            pl.plot(T, self.df.values[:, i], color = 'tab:red')
            pl.scatter(np.array(T)[self.result],
                       self.df.values[self.result, i],
                       s = 80,
                       facecolors = 'none',
                       edgecolors = 'b')
            pl.gca().set(ylabel = r'$' + str(self.df.columns.values[i]) + '$')
        if show:
            pl.show()