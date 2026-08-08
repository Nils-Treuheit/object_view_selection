class FilterPipeline:

    def __init__(self, filters):
        self.filters = filters

    @property
    def need_fitting(self):
        """True when any wrapped filter needs a population pass (outlier mode)."""
        return any(
            getattr(f, "need_fitting", lambda: False)()
            for f in self.filters
        )

    def fit_observations(self, observations):
        """One population pass before the main loop for outlier-based filters."""
        for f in self.filters:
            if getattr(f, "need_fitting", lambda: False)():
                f.fit(observations)

    def run(self, observation):
        for f in self.filters:
            score, passed, reason = f.evaluate(observation)
            if not passed:
                observation.rejected = True
                observation.rejection_reason = reason
                return False
        return True
