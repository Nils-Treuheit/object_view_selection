from abc import ABC, abstractmethod

from data_io.observation import Observation


class BaseFilter(ABC):

    def __init__(self, enabled=True):

        self.enabled = enabled

    @abstractmethod
    def evaluate(self, observation: Observation):
        """
        Returns
        -------
        score : float
            Normalized score in [0,1]

        passed : bool
            Whether observation passes

        reason : str
            Reason if rejected
        """
        pass
