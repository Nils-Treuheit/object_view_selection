class QualityScorer:

    def __init__(self, metrics, weights):
        self.metrics = metrics
        self.weights = weights

    def score(self, observation):

        total = 0.0
        wsum = 0.0

        for metric in self.metrics:
            w = self.weights.get(metric.name, 1.0)
            s = metric.compute(observation)
            setattr(observation.metrics, metric.name, s)
            total += w * s
            wsum += w

        observation.quality = total / wsum
        return observation.quality