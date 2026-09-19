"""Errors that stop a command with a message for the operator."""


class ModebenchError(Exception):
    """Base class of every error that the command line reports without a traceback."""


class ConfigError(ModebenchError):
    """A configuration file is missing, malformed or inconsistent."""


class DatasetError(ModebenchError):
    """A dataset file is missing, malformed or inconsistent."""


class PrivacyViolation(ModebenchError):
    """A private dataset would go to a route that can keep or train on the data."""


class CostCeilingExceeded(ModebenchError):
    """The estimated or the actual cost of a run is above the ceiling."""


class PreflightError(ModebenchError):
    """A key is missing or a model identifier is not in the catalogue of its provider."""


class StorageError(ModebenchError):
    """A run is not in the database, or the database is not usable."""
