from .variants import FilterVariant


class FilterPipeline:

    def __init__(self, filters):
        self.filters = filters

    @property
    def requires_fit(self):
        """True when any wrapped filter needs a population pass (outlier mode)."""
        return any(
            getattr(f, "requires_fit", lambda: False)()
            if isinstance(f, FilterVariant)
            else False
            for f in self.filters
        )

    def fit_observations(self, observations):
        """One population pass before the main loop for outlier-based filters."""
        for f in self.filters:
            if isinstance(f, FilterVariant):
                f.fit(observations)

    def run(self, observation):
        for f in self.filters:
            score, passed, reason = f.evaluate(observation)
            if not passed:
                observation.rejected = True
                observation.rejection_reason = reason
                return False
        return True
