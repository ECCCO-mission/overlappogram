class OverlappogramWarning(Warning):
    pass


class NoWeightsWarnings(OverlappogramWarning):
    """There are no weights passed to unfold."""
