"""Normal score transform and back-transform (GSLIB nscore / backtr)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class NormalScoreTransform:
    """Fitted transform table: sorted data values <-> normal scores."""
    values: np.ndarray   # sorted original values
    scores: np.ndarray   # matching normal scores
    zmin: float
    zmax: float

    def forward(self, x):
        raise NotImplementedError

    def inverse(self, y, ltail="linear", utail="linear",
                ltpar=None, utpar=None):
        raise NotImplementedError


def nscore(x, wts=None, zmin=None, zmax=None):
    """Weighted normal score transform. Returns (scores, NormalScoreTransform)."""
    raise NotImplementedError


def backtr(y, transform, ltail="linear", utail="linear",
           ltpar=None, utpar=None):
    """Back-transform normal scores using a fitted ``NormalScoreTransform``."""
    raise NotImplementedError
