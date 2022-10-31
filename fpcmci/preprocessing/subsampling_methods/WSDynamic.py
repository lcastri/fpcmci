import ruptures as rpt
from .EntropyBasedMethod import EntropyBasedMethod
from .SubsamplingMethod import SubsamplingMethod, SSMode


class EntropyBasedDynamic(SubsamplingMethod, EntropyBasedMethod):
    def __init__(self, window_min_size, entropy_threshold):
        """
        Subsampling method with static window size based on Fourier analysis

        Args:
            window_min_size (int): minimun window size
            entropy_threshold (float): entropy threshold
        """
        SubsamplingMethod.__init__(self, SSMode.WSDynamic)
        EntropyBasedMethod.__init__(self, entropy_threshold)
        if window_min_size is None:
            raise ValueError("window_type = DYNAMIC but window_min_size not specified")
        self.wms = window_min_size
        self.ws = None

    def dataset_segmentation(self):
        """
        Segments dataset based on breakpoint analysis and a min window size
        """
        de = self.create_rounded_copy()
        algo = rpt.Pelt(model = "l2", min_size = self.wms).fit(de)
        seg_res = algo.predict(pen = 10)
        self.segments = [(seg_res[i - 1], seg_res[i]) for i in range(1, len(seg_res))]
        self.segments.insert(0, (0, seg_res[0]))


    def run(self):
        # build list of segment
        self.dataset_segmentation()

        # compute entropy moving window
        self.moving_window_analysis()

        # extracting subsampling procedure results
        idxs = self.extract_indexes()

        return idxs