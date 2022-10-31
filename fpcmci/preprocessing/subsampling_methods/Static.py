from .SubsamplingMethod import SubsamplingMethod, SSMode


class Static(SubsamplingMethod):
    def __init__(self, step):
        """
        subsample data by taking one sample each step-samples

        Args:
            step (int): step
        """
        super().__init__(SSMode.Static)
        if step is None:
            raise ValueError("step not specified")
        self.step = step

    def run(self):
        return range(0, len(self.df.values), self.step)