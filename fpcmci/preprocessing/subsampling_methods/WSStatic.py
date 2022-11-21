from fpcmci.preprocessing.subsampling_methods.EntropyBasedMethod import EntropyBasedMethod
from fpcmci.preprocessing.subsampling_methods.SubsamplingMethod import SubsamplingMethod, SSMode


class EntropyBasedStatic(SubsamplingMethod, EntropyBasedMethod):
    def __init__(self, window_size, entropy_threshold):
        """
        Entropy based subsampling method with static window size

        Args:
            window_min_size (int): minimun window size
            entropy_threshold (float): entropy threshold
        """
        
        SubsamplingMethod.__init__(self, SSMode.WSDynamic)
        EntropyBasedMethod.__init__(self, entropy_threshold)
        if window_size is None:
            raise ValueError("window_type = STATIC but window_size not specified")
        self.ws = window_size


    def dataset_segmentation(self):
        """
        Segments dataset with a fixed window size
        """
        seg_res = [i for i in range(0, len(self.df.values), self.ws)]
        self.segments = [(i, i + self.ws) for i in range(0, len(self.df.values) - self.ws, self.ws)]
        if not seg_res.__contains__(len(self.df.values)):
            self.segments.append((seg_res[-1], len(self.df.values)))
            seg_res.append(len(self.df.values))


    def run(self):
        # build list of segment
        self.dataset_segmentation()

        # compute entropy moving window
        self.moving_window_analysis()

        # extracting subsampling procedure results
        idxs = self.extract_indexes()

        return idxs