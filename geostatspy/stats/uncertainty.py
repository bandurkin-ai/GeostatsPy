"""Bootstrap and local (per-location) statistics over realization ensembles."""


def bootstrap(data, stat, n_boot=1000, seed=None):
    """Bootstrap distribution of ``stat(sample)``."""
    raise NotImplementedError


def local_expectation(realizations):
    """Mean across realizations (axis 0)."""
    raise NotImplementedError


def local_std(realizations):
    """Standard deviation across realizations (axis 0)."""
    raise NotImplementedError


def local_percentile(realizations, p):
    """p-th percentile (0-100) across realizations (axis 0)."""
    raise NotImplementedError


def local_prob_exceed(realizations, threshold):
    """Probability of exceeding ``threshold`` across realizations (axis 0)."""
    raise NotImplementedError
