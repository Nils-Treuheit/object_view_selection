class QualityScorer:
"""
Usage:
scorer = QualityScorer(

    metrics=[

        BlurQuality(),

        AreaQuality(),

        OcclusionQuality(),

        CompletenessQuality(),

    ],

    weights={

        "blur":0.3,

        "area":0.2,

        "occlusion":0.2,

        "completeness":0.3,

    }

)
"""

    def __init__(

        self,

        metrics,

        weights,

    ):

        self.metrics = metrics

        self.weights = weights

    def score(

        self,

        observation,

    ):

        total = 0.0

        wsum = 0.0

        for metric in self.metrics:

            w = self.weights.get(
                metric.name,
                1.0,
            )

            s = metric.compute(observation)

            observation.metrics[
                metric.name
            ] = s

            total += w * s

            wsum += w

        observation.quality = total / wsum

        return observation.quality
