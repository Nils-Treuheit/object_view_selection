from abc import ABC, abstractmethod

from io.observation import Observation


class QualityMetric(ABC):

    name = "metric"

    @abstractmethod
    def compute(self, observation: Observation) -> float:
        """
        Returns
        -------
        float
            Normalized score in [0,1]
        """
        pass
