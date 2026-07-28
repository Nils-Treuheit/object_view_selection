import numpy as np


class MinMaxNormalizer:

    def __init__(self, minimum, maximum):

        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, value):

        value = (value - self.minimum) / (
            self.maximum - self.minimum
        )

        return float(np.clip(value, 0.0, 1.0))
