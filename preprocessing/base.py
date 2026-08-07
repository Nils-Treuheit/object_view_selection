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
        
        Each filter has to implement a absolute threshold-based rejection criteria to filter out
        complete unusable garbage and unless decision is binary (e.g. touches border [truncation], 
        no object mask [empty]) the filter also has to implement an population-based rejection of 
        noticeably (by a large margin) bad outliers       
        """
        pass
