"""Exceptions raised by the registry."""


class RegistryError(Exception):
    """Base class for all registry-related errors."""


class UnknownMitigationError(KeyError):
    """Raised when a mitigation name is not in the merged registry."""
