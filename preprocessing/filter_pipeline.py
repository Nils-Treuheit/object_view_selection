class FilterPipeline:

    def __init__(self, filters):

        self.filters = filters

    def run(self, observation):

        for f in self.filters:

            score, passed, reason = f.evaluate(observation)

            observation.metrics[
                f.__class__.__name__
            ] = score

            if not passed:

                observation.rejected = True

                observation.rejection_reason = reason

                return False

        return True
