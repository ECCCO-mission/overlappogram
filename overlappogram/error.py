class OverlappogramWarning(Warning):
    pass


class NoWeightsWarnings(OverlappogramWarning):
    """There are no weights passed to unfold."""


class OverlappogramError(Exception):
    pass


class InvalidDataFormatError(OverlappogramError):
    """The data file has an error that prevents processing."""


class InvalidInversionModeError(OverlappogramError):
    """The inversion mode is not allowed."""
